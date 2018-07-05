import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
from torchvision.utils import save_image
from source.dis_model.discriminator import Discriminator, SNDiscriminator, SNResNetProjectionDiscriminator
from source.gen_model.generator import *
from updater import *
from PIL import Image

class Solver(object):

    def __init__(self, dataset1_loader, dataset2_loader, config):
        self.device_ids = config.device_ids

        # Data loader
        self.dataset1_loader = dataset1_loader
        self.dataset2_loader = dataset2_loader

        # Model hyper-parameters
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.d_train_repeat = config.d_train_repeat


        # Hyper-parameteres
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_reg = config.lambda_reg

        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        # self.dataset = config.dataset
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.batch_size = config.batch_size
        self.use_tensorboard = config.use_tensorboard
        self.pretrained_model = config.pretrained_model
        self.resume = config.resume

        # Test settings
        self.test_model = config.test_model
        self.test_interval = config.test_interval
        self.test_iters = config.test_iters
        self.test_step = config.test_step
        self.test_traverse = config.test_traverse
        self.test_save_path = os.path.join(config.task_name, config.evaluates)

        # Path
        self.log_path = os.path.join(config.ckpt_path, config.task_name, config.log_path)
        self.sample_path = os.path.join(config.ckpt_path, config.task_name, config.sample_path)
        self.model_save_path = os.path.join(config.ckpt_path, config.task_name, config.model_save_path)
        self.result_path = os.path.join(config.ckpt_path, config.task_name, config.result_path)
        self.test_label_path = config.test_label_path

        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.model_save_star = config.model_save_star

        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.resume:
            self.load_pretrained_model()
        elif self.pretrained_model:
            self.load_pretrained_diff_model()

    def build_model(self):
        # Define a generator and a discriminator
        # self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        # self.G = Generator_small(self.g_conv_dim, self.c_dim, self.g_repeat_num, identity=False, self_attention=False)
        # self.G = Generator_FNT(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        # self.G = Generator_FNT_mutiviewsample(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        # self.G = Generator_small_mutiviewsample(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.G = Generator_small_SelfAttention(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        # self.D = Discriminator(self.d_conv_dim, self.c_dim)
        # self.D = SNDiscriminator(self.d_conv_dim, self.c_dim)
        self.D = SNResNetProjectionDiscriminator(self.d_conv_dim, ch_in=3, n_classes=self.c_dim)
        self.G = torch.nn.DataParallel(self.G, device_ids=self.device_ids)
        self.D = torch.nn.DataParallel(self.D, device_ids=self.device_ids)

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        # Print networks
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.resume))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.resume))))
        print('loaded trained models (step: {})..!'.format(self.resume))

    
    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        x = Variable(x, requires_grad=requires_grad)
        return x

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def threshold(self, x):
        x = x.clone()
        x = (x >= 0.5).float()
        # x[x >= 0.5] = 1
        # x[x < 0.5] = 0
        return x

    def compute_accuracy(self, x, y, n_classes=5):
        #todo:
        if x.dim() > 1:
            _, predicted = torch.max(x, dim=1)
            correct = (predicted == y).float()
        else:
            eps = 1./n_classes/4.
            n_intervals = n_classes - 1
            target = y.float()/(n_intervals*0.5)-1.
            correct = ( (x >= target-eps) * (x < target+eps ) ).float()
        accuracy = torch.mean(correct) * 100.0
        return accuracy

    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def train(self):
        """Train StarGAN within a single dataset."""

        # Set dataloader
        self.data_loader = self.dataset1_loader

        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader)

        fixed_x = []
        real_c = []
        for i, (images, labels) in enumerate(self.data_loader):
            fixed_x.append(images)
            real_c.append(labels)
            if i == 0:
                break

        # Fixed inputs and target domain labels for debugging
        fixed_x = torch.cat(fixed_x, dim=0)
        fixed_x = self.to_var(fixed_x, requires_grad=False)
        real_c = torch.cat(real_c, dim=0)

        fixed_c_list = []
        for i in range(self.c_dim):
            fixed_c = torch.ones(fixed_x.size(0)) * i
            fixed_c_list.append(self.to_var(fixed_c, requires_grad=False))

        # lr cache for decaying
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start with trained model if exists
        if self.resume:
            start = int(self.resume.split('_')[0])
        else:
            start = 0

        # Start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            for i, (real_x, real_label) in enumerate(self.data_loader):
                
                # Generat fake labels randomly (target domain labels)
                rand_idx = torch.randperm(real_label.size(0))
                fake_label = real_label[rand_idx]

                real_c = self.one_hot(real_label, self.c_dim)
                fake_c = self.one_hot(fake_label, self.c_dim)

                # Convert tensor to variable
                real_x = self.to_var(real_x)
                real_c = self.to_var(real_c)           # input for the generator
                fake_c = self.to_var(fake_c)
                real_label = self.to_var(real_label)   # this is same as real_c if dataset == 'CelebA'
                fake_label = self.to_var(fake_label)
                
                # ================== Train D ================== #

                # Compute loss with real images
                out_real, out_reg = self.D(real_x, real_label)
                # Compute loss with fake images
                fake_x = self.G(real_x, fake_label).detach()
                out_fake, _ = self.D(fake_x, fake_label)

                d_loss_adv = loss_hinge_dis(out_fake, out_real)
                # todo:regression
                d_loss_reg = loss_hard_reg(out_reg, real_label, self.c_dim)

                # Compute classification accuracy of the discriminator
                if (i+1) % self.log_step == 0:
                    classification_accuracies = self.compute_accuracy(out_real, real_label, n_classes=self.c_dim)
                    regression_accuracies = self.compute_accuracy(out_reg.squeeze(), real_label, n_classes=self.c_dim)

                    log = "{:.2f}/{:.2f}".format(classification_accuracies.data.cpu().numpy(), regression_accuracies.data.cpu().numpy())
                    print('Classification/regression Acc: ', end='')
                    print(log)


                # Backward + Optimize
                d_loss = d_loss_adv + self.lambda_reg * d_loss_reg
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging
                loss = {}
                loss['D/loss_adv'] = d_loss_adv.data
                loss['D/loss_reg'] = d_loss_reg.data

                # ================== Train G ================== #
                if (i+1) % self.d_train_repeat == 0:

                    # Original-to-target and target-to-original domain
                    fake_x = self.G(real_x, fake_label)
                    # todo
                    rec_x  = self.G(real_x, real_label)

                    # Compute losses
                    out_fake, out_reg = self.D(fake_x, fake_label)
                    g_loss_adv = loss_hinge_gen(out_fake)
                    g_loss_rec = torch.mean(torch.abs(real_x - rec_x))
                    g_loss_reg = loss_hard_reg(out_reg, fake_label, self.c_dim)

                    # Backward + Optimize
                    g_loss = g_loss_adv + self.lambda_rec * g_loss_rec + self.lambda_reg * g_loss_reg
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    loss['G/loss_adv'] = g_loss_adv.data
                    loss['G/loss_rec'] = g_loss_rec.data
                    loss['G/loss_reg'] = g_loss_reg.data

                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e+1, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

                # Translate fixed images for debugging
                if (i+1) % self.sample_step == 0:
                    fake_image_list = [fixed_x]
                    for fixed_c in fixed_c_list:
                        fake_image_list.append(self.G(fixed_x, fixed_c).detach())
                    fake_images = torch.cat(fake_image_list, dim=3)
                    save_image(self.denorm(fake_images.data),
                        os.path.join(self.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)
                    print('Translated images and saved into {}..!'.format(self.sample_path))

            # Save model checkpoints
            if (e+1) % self.model_save_step == 0 and (e+1) > self.model_save_star:
                print('Save model checkpoints')
                torch.save(self.G.state_dict(),
                    os.path.join(self.model_save_path, '{}_G.pth'.format(e+1)))
                torch.save(self.D.state_dict(),
                    os.path.join(self.model_save_path, '{}_D.pth'.format(e+1)))

            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def train_multi(self):
        """Train StarGAN with multiple datasets.
        In the code below, 1 is related to CelebA and 2 is releated to RaFD.
        """
        # Fixed imagse and labels for debugging
        fixed_x1 = []
        fixed_x2 = []
        real_c = []

        for i, (images, labels) in enumerate(self.dataset1_loader):
            fixed_x1.append(images)
            real_c.append(labels)
            if i == 2:
                break

        for i, (images, labels) in enumerate(self.dataset2_loader):
            fixed_x2.append(images)
            # real_c.append(labels)
            if i == 2:
                break

        fixed_x1 = torch.cat(fixed_x1, dim=0)
        fixed_x1 = self.to_var(fixed_x1, volatile=True)
        fixed_x2 = torch.cat(fixed_x2, dim=0)
        fixed_x2 = self.to_var(fixed_x2, volatile=True)
        real_c = torch.cat(real_c, dim=0)
        fixed_c1_list = self.make_celeb_labels(real_c)

        fixed_c2_list = []
        for i in range(self.c2_dim):
            fixed_c = self.one_hot(torch.ones(fixed_x1.size(0)) * i, self.c2_dim)
            fixed_c2_list.append(self.to_var(fixed_c, volatile=True))

        fixed_zero1 = self.to_var(torch.zeros(fixed_x1.size(0), self.c2_dim))     # zero vector when training with CelebA
        fixed_mask1 = self.to_var(self.one_hot(torch.zeros(fixed_x1.size(0)), 2)) # mask vector: [1, 0]
        fixed_zero2 = self.to_var(torch.zeros(fixed_x1.size(0), self.c_dim))      # zero vector when training with RaFD
        fixed_mask2 = self.to_var(self.one_hot(torch.ones(fixed_x1.size(0)), 2))  # mask vector: [0, 1]

        # lr cache for decaying
        g_lr = self.g_lr
        d_lr = self.d_lr

        # data iterator
        data_iter1 = iter(self.dataset1_loader)
        data_iter2 = iter(self.dataset2_loader)

        # Start with trained model
        if self.resume:
            start = int(self.resume) + 1
        else:
            start = 0

        # # Start training
        start_time = time.time()
        for i in range(start, self.num_iters):

            # Fetch mini-batch images and labels
            try:
                real_x1, real_label1 = next(data_iter1)
            except:
                data_iter1 = iter(self.dataset1_loader)
                real_x1, real_label1 = next(data_iter1)

            try:
                real_x2, real_label2 = next(data_iter2)
            except:
                data_iter2 = iter(self.dataset2_loader)
                real_x2, real_label2 = next(data_iter2)

            # Generate fake labels randomly (target domain labels)
            rand_idx = torch.randperm(real_label1.size(0))
            fake_label1 = real_label1[rand_idx]
            rand_idx = torch.randperm(real_label2.size(0))
            fake_label2 = real_label2[rand_idx]

            real_c1 = real_label1.clone()
            fake_c1 = fake_label1.clone()
            zero1 = torch.zeros(real_x1.size(0), self.c2_dim)
            mask1 = self.one_hot(torch.zeros(real_x1.size(0)), 2)

            real_c2 = self.one_hot(real_label2, self.c2_dim)
            fake_c2 = self.one_hot(fake_label2, self.c2_dim)
            zero2 = torch.zeros(real_x2.size(0), self.c_dim)
            mask2 = self.one_hot(torch.ones(real_x2.size(0)), 2)

            # Convert tensor to variable
            real_x1 = self.to_var(real_x1)
            real_c1 = self.to_var(real_c1)
            fake_c1 = self.to_var(fake_c1)
            mask1 = self.to_var(mask1)
            zero1 = self.to_var(zero1)

            real_x2 = self.to_var(real_x2)
            real_c2 = self.to_var(real_c2)
            fake_c2 = self.to_var(fake_c2)
            mask2 = self.to_var(mask2)
            zero2 = self.to_var(zero2)

            real_label1 = self.to_var(real_label1)
            fake_label1 = self.to_var(fake_label1)
            real_label2 = self.to_var(real_label2)
            fake_label2 = self.to_var(fake_label2)

            # ================== Train D ================== #

            # Real images (CelebA)
            out_real, out_cls = self.D(real_x1)
            out_cls1 = out_cls[:, :self.c_dim]      # celebA part
            d_loss_real = - torch.mean(out_real)
            d_loss_cls = F.binary_cross_entropy_with_logits(out_cls1, real_label1, size_average=False) / real_x1.size(0)

            # Real images (RaFD)
            out_real, out_cls = self.D(real_x2)
            out_cls2 = out_cls[:, self.c_dim:]      # rafd part
            d_loss_real += - torch.mean(out_real)
            d_loss_cls += F.cross_entropy(out_cls2, real_label2)

            # Compute classification accuracy of the discriminator
            if (i+1) % self.log_step == 0:
                accuracies = self.compute_accuracy(out_cls1, real_label1, 'CelebA')
                log = ["{:.2f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                print('Classification Acc (Black/Blond/Brown/Gender/Aged): ', end='')
                print(log)
                accuracies = self.compute_accuracy(out_cls2, real_label2, 'RaFD')
                log = ["{:.2f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                print('Classification Acc (8 emotional expressions): ', end='')
                print(log)

            # Fake images (CelebA)
            fake_c = torch.cat([fake_c1, zero1, mask1], dim=1)
            fake_x1 = self.G(real_x1, fake_c)
            fake_x1 = Variable(fake_x1.data)
            out_fake, _ = self.D(fake_x1)
            d_loss_fake = torch.mean(out_fake)

            # Fake images (RaFD)
            fake_c = torch.cat([zero2, fake_c2, mask2], dim=1)
            fake_x2 = self.G(real_x2, fake_c)
            out_fake, _ = self.D(fake_x2)
            d_loss_fake += torch.mean(out_fake)

            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Compute gradient penalty
            if (i+1) % 2 == 0:
                real_x = real_x1
                fake_x = fake_x1
            else:
                real_x = real_x2
                fake_x = fake_x2

            alpha = torch.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real_x)
            interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
            out, out_cls = self.D(interpolated)

            if (i+1) % 2 == 0:
                out_cls = out_cls[:, :self.c_dim]  # CelebA
            else:
                out_cls = out_cls[:, self.c_dim:]  # RaFD

            grad = torch.autograd.grad(outputs=out,
                                       inputs=interpolated,
                                       grad_outputs=torch.ones(out.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1)**2)

            # Backward + Optimize
            d_loss = self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging
            loss = {}
            loss['D/loss_real'] = d_loss_real.data[0]
            loss['D/loss_fake'] = d_loss_fake.data[0]
            loss['D/loss_cls'] = d_loss_cls.data[0]
            loss['D/loss_gp'] = d_loss_gp.data[0]

            # ================== Train G ================== #
            if (i+1) % self.d_train_repeat == 0:
                # Original-to-target and target-to-original domain (CelebA)
                fake_c = torch.cat([fake_c1, zero1, mask1], dim=1)
                real_c = torch.cat([real_c1, zero1, mask1], dim=1)
                fake_x1 = self.G(real_x1, fake_c)
                rec_x1 = self.G(fake_x1, real_c)

                # Compute losses
                out, out_cls = self.D(fake_x1)
                out_cls1 = out_cls[:, :self.c_dim]
                g_loss_fake = - torch.mean(out)
                g_loss_rec = torch.mean(torch.abs(real_x1 - rec_x1))
                g_loss_cls = F.binary_cross_entropy_with_logits(out_cls1, fake_label1, size_average=False) / fake_x1.size(0)

                # Original-to-target and target-to-original domain (RaFD)
                fake_c = torch.cat([zero2, fake_c2, mask2], dim=1)
                real_c = torch.cat([zero2, real_c2, mask2], dim=1)
                fake_x2 = self.G(real_x2, fake_c)
                rec_x2 = self.G(fake_x2, real_c)

                # Compute losses
                out, out_cls = self.D(fake_x2)
                out_cls2 = out_cls[:, self.c_dim:]
                g_loss_fake += - torch.mean(out)
                g_loss_rec += torch.mean(torch.abs(real_x2 - rec_x2))
                g_loss_cls += F.cross_entropy(out_cls2, fake_label2)

                # Backward + Optimize
                g_loss = g_loss_fake + self.lambda_cls * g_loss_cls + self.lambda_rec * g_loss_rec
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging
                loss['G/loss_fake'] = g_loss_fake.data[0]
                loss['G/loss_cls'] = g_loss_cls.data[0]
                loss['G/loss_rec'] = g_loss_rec.data[0]

            # Print out log info
            if (i+1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))

                log = "Elapsed [{}], Iter [{}/{}]".format(
                    elapsed, i+1, self.num_iters)

                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate the images (debugging)
            if (i+1) % self.sample_step == 0:
                fake_image_list = [fixed_x1]
                self.G.eval()
                # Changing hair color, gender, and age
                for j in range(self.c_dim):
                    fake_c = torch.cat([fixed_c1_list[j], fixed_zero1, fixed_mask1], dim=1)
                    fake_image_list.append(self.G(fixed_x1, fake_c))
                # Changing emotional expressions
                for j in range(self.c2_dim):
                    fake_c = torch.cat([fixed_zero2, fixed_c2_list[j], fixed_mask2], dim=1)
                    fake_image_list.append(self.G(fixed_x1, fake_c))
                fake = torch.cat(fake_image_list, dim=3)

                # Save the translated images
                save_image(self.denorm(fake.data),
                    os.path.join(self.sample_path, '{}_fake_x1.png'.format(i+1)), nrow=1, padding=0)

                fake_image_list = [fixed_x2]
                self.G.eval()
                # Changing hair color, gender, and age
                for j in range(self.c_dim):
                    fake_c = torch.cat([fixed_c1_list[j], fixed_zero1, fixed_mask1], dim=1)
                    fake_image_list.append(self.G(fixed_x2, fake_c))
                # Changing emotional expressions
                for j in range(self.c2_dim):
                    fake_c = torch.cat([fixed_zero2, fixed_c2_list[j], fixed_mask2], dim=1)
                    fake_image_list.append(self.G(fixed_x2, fake_c))
                fake = torch.cat(fake_image_list, dim=3)

                self.G.train()
                # Save the translated images
                save_image(self.denorm(fake.data),
                           os.path.join(self.sample_path, '{}_fake_x2.png'.format(i + 1)), nrow=1, padding=0)


            # Save model checkpoints
            if (i+1) % self.model_save_step == 0:
                torch.save(self.G.state_dict(),
                    os.path.join(self.model_save_path, '{}_G.pth'.format(i+1)))
                torch.save(self.D.state_dict(),
                    os.path.join(self.model_save_path, '{}_D.pth'.format(i+1)))

            # Decay learning rate
            decay_step = 1000
            if (i+1) > (self.num_iters - self.num_iters_decay) and (i+1) % decay_step==0:
                g_lr -= (self.g_lr / float(self.num_iters_decay) * decay_step)
                d_lr -= (self.d_lr / float(self.num_iters_decay) * decay_step)
                self.update_lr(g_lr, d_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def train_multi_branch(self):
        """Train StarGAN with multiple datasets.
        In the code below, 1 is related to CelebA and 2 is releated to RaFD.
        """
        # Fixed imagse and labels for debugging
        fixed_x1 = []
        fixed_x2 = []
        real_c = []

        for i, (images, labels) in enumerate(self.dataset1_loader):
            fixed_x1.append(images)
            real_c.append(labels)
            if i == 2:
                break

        for i, (images, labels) in enumerate(self.dataset2_loader):
            fixed_x2.append(images)
            # real_c.append(labels)
            if i == 2:
                break

        fixed_x1 = torch.cat(fixed_x1, dim=0)
        fixed_x1 = self.to_var(fixed_x1, volatile=True)
        fixed_x2 = torch.cat(fixed_x2, dim=0)
        fixed_x2 = self.to_var(fixed_x2, volatile=True)
        real_c = torch.cat(real_c, dim=0)
        fixed_c1_list = self.make_celeb_labels(real_c)

        fixed_c2_list = []
        for i in range(self.c2_dim):
            fixed_c = self.one_hot(torch.ones(fixed_x1.size(0)) * i, self.c2_dim)
            fixed_c2_list.append(self.to_var(fixed_c, volatile=True))

        # fixed_zero1 = self.to_var(torch.zeros(fixed_x1.size(0), self.c2_dim))     # zero vector when training with CelebA
        # fixed_mask1 = self.to_var(self.one_hot(torch.zeros(fixed_x1.size(0)), 2)) # mask vector: [1, 0]
        # fixed_zero2 = self.to_var(torch.zeros(fixed_x1.size(0), self.c_dim))      # zero vector when training with RaFD
        # fixed_mask2 = self.to_var(self.one_hot(torch.ones(fixed_x1.size(0)), 2))  # mask vector: [0, 1]

        # lr cache for decaying
        g_lr = self.g_lr
        d_lr = self.d_lr

        # data iterator
        data_iter1 = iter(self.dataset1_loader)
        data_iter2 = iter(self.dataset2_loader)

        # Start with trained model
        if self.resume:
            start = int(self.resume) + 1
        else:
            start = 0

        # # Start training
        start_time = time.time()
        for i in range(start, self.num_iters):

            # Fetch mini-batch images and labels
            if i <= self.stage1:
                try:
                    real_x1, real_label1 = next(data_iter1)
                except:
                    data_iter1 = iter(self.dataset1_loader)
                    real_x1, real_label1 = next(data_iter1)

            try:
                real_x2, real_label2 = next(data_iter2)
            except:
                data_iter2 = iter(self.dataset2_loader)
                real_x2, real_label2 = next(data_iter2)

            # Generate fake labels randomly (target domain labels)
            if i <= self.stage1:
                rand_idx = torch.randperm(real_label1.size(0))
                fake_label1 = real_label1[rand_idx]
            rand_idx = torch.randperm(real_label2.size(0))
            fake_label2 = real_label2[rand_idx]

            if i <= self.stage1:
                real_c1 = real_label1.clone()
                fake_c1 = fake_label1.clone()
            # zero1 = torch.zeros(real_x1.size(0), self.c2_dim)
            # mask1 = self.one_hot(torch.zeros(real_x1.size(0)), 2)

            real_c2 = self.one_hot(real_label2, self.c2_dim)
            fake_c2 = self.one_hot(fake_label2, self.c2_dim)
            # zero2 = torch.zeros(real_x2.size(0), self.c_dim)
            # mask2 = self.one_hot(torch.ones(real_x2.size(0)), 2)

            # Convert tensor to variable
            if i <= self.stage1:
                real_x1 = self.to_var(real_x1)
                real_c1 = self.to_var(real_c1)
                fake_c1 = self.to_var(fake_c1)
            # mask1 = self.to_var(mask1)
            # zero1 = self.to_var(zero1)

            real_x2 = self.to_var(real_x2)
            real_c2 = self.to_var(real_c2)
            fake_c2 = self.to_var(fake_c2)
            # mask2 = self.to_var(mask2)
            # zero2 = self.to_var(zero2)

            if i <= self.stage1:
                real_label1 = self.to_var(real_label1)
                fake_label1 = self.to_var(fake_label1)
            real_label2 = self.to_var(real_label2)
            fake_label2 = self.to_var(fake_label2)

            # ================== Train D ================== #

            # Real images (RaFD)
            out_real, out_cls = self.D(real_x2)
            out_cls2 = out_cls[:, self.c_dim:]  # rafd part
            d_loss_real = - torch.mean(out_real)
            d_loss_cls = F.cross_entropy(out_cls2, real_label2)

            # Real images (CelebA)
            if i <= self.stage1:
                out_real, out_cls = self.D(real_x1)
                out_cls1 = out_cls[:, :self.c_dim]      # celebA part
                d_loss_real += - torch.mean(out_real)
                d_loss_cls += F.binary_cross_entropy_with_logits(out_cls1, real_label1, size_average=False) / real_x1.size(0)

            # Compute classification accuracy of the discriminator
            if (i+1) % self.log_step == 0:
                if i <= self.stage1:
                    accuracies = self.compute_accuracy(out_cls1, real_label1, 'CelebA')
                    log = ["{:.2f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                    print('Classification Acc (Black/Blond/Brown/Gender/Aged): ', end='')
                    print(log)
                accuracies = self.compute_accuracy(out_cls2, real_label2, 'RaFD')
                log = ["{:.2f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                print('Classification Acc (8 emotional expressions): ', end='')
                print(log)

            # Fake images (RaFD)
            # fake_c = torch.cat([zero2, fake_c2, mask2], dim=1)
            fake_x2 = self.G(real_x2, fake_c2, branch_id=2)
            out_fake, _ = self.D(fake_x2)
            d_loss_fake = torch.mean(out_fake)

            # Fake images (CelebA)
            # fake_c = torch.cat([fake_c1, zero1, mask1], dim=1)
            if i <= self.stage1:
                fake_x1 = self.G(real_x1, fake_c1, branch_id=1)
                fake_x1 = Variable(fake_x1.data)
                out_fake, _ = self.D(fake_x1)
                d_loss_fake += torch.mean(out_fake)

            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Compute gradient penalty
            if (i+1) % 2 == 0 and i <= self.stage1:
                real_x = real_x1
                fake_x = fake_x1
            else:
                real_x = real_x2
                fake_x = fake_x2

            alpha = torch.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real_x)
            interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
            out, out_cls = self.D(interpolated)

            if (i+1) % 2 == 0 and i <= self.stage1:
                out_cls = out_cls[:, :self.c_dim]  # CelebA
            else:
                out_cls = out_cls[:, self.c_dim:]  # RaFD

            grad = torch.autograd.grad(outputs=out,
                                       inputs=interpolated,
                                       grad_outputs=torch.ones(out.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1)**2)

            # Backward + Optimize
            d_loss = self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging
            loss = {}
            loss['D/loss_real'] = d_loss_real.data[0]
            loss['D/loss_fake'] = d_loss_fake.data[0]
            loss['D/loss_cls'] = d_loss_cls.data[0]
            loss['D/loss_gp'] = d_loss_gp.data[0]

            # ================== Train G ================== #
            if (i+1) % self.d_train_repeat == 0:
                # Original-to-target and target-to-original domain (RaFD)
                # fake_c = torch.cat([zero2, fake_c2, mask2], dim=1)
                # real_c = torch.cat([zero2, real_c2, mask2], dim=1)
                fake_x2 = self.G(real_x2, fake_c2, branch_id=2)
                rec_x2 = self.G(fake_x2, real_c2, branch_id=2)

                # Compute losses
                out, out_cls = self.D(fake_x2)
                out_cls2 = out_cls[:, self.c_dim:]
                g_loss_fake = - torch.mean(out)
                g_loss_rec = torch.mean(torch.abs(real_x2 - rec_x2))
                g_loss_cls = F.cross_entropy(out_cls2, fake_label2)

                # Original-to-target and target-to-original domain (CelebA)
                # fake_c = torch.cat([fake_c1, zero1, mask1], dim=1)
                # real_c = torch.cat([real_c1, zero1, mask1], dim=1)
                if i <= self.stage1:
                    fake_x1 = self.G(real_x1, fake_c1, branch_id=1)
                    rec_x1 = self.G(fake_x1, real_c1, branch_id=1)

                    # Compute losses
                    out, out_cls = self.D(fake_x1)
                    out_cls1 = out_cls[:, :self.c_dim]
                    g_loss_fake += - torch.mean(out)
                    g_loss_rec += torch.mean(torch.abs(real_x1 - rec_x1))
                    g_loss_cls += F.binary_cross_entropy_with_logits(out_cls1, fake_label1, size_average=False) / fake_x1.size(0)



                # Backward + Optimize
                g_loss = g_loss_fake + self.lambda_cls * g_loss_cls + self.lambda_rec * g_loss_rec
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging
                loss['G/loss_fake'] = g_loss_fake.data[0]
                loss['G/loss_cls'] = g_loss_cls.data[0]
                loss['G/loss_rec'] = g_loss_rec.data[0]

            # Print out log info
            if (i+1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))

                log = "Elapsed [{}], Iter [{}/{}]".format(
                    elapsed, i+1, self.num_iters)

                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate the images (debugging)
            if (i+1) % self.sample_step == 0:
                fake_image_list = [fixed_x1]
                self.G.eval()

                if i <= self.stage1:
                    for j in range(self.c_dim):
                        # fake_c = torch.cat([fixed_c1_list[j], fixed_zero1, fixed_mask1], dim=1)
                        fake_image_list.append(self.G(fixed_x1, fixed_c1_list[j], branch_id=1))
                    # Changing emotional expressions
                    for j in range(self.c2_dim):
                        # fake_c = torch.cat([fixed_zero2, fixed_c2_list[j], fixed_mask2], dim=1)
                        fake_image_list.append(self.G(fixed_x1, fixed_c2_list[j], branch_id=2))
                    fake = torch.cat(fake_image_list, dim=3)

                    # Save the translated images
                    save_image(self.denorm(fake.data),
                        os.path.join(self.sample_path, '{}_fake_x1.png'.format(i+1)), nrow=1, padding=0)

                fake_image_list = [fixed_x2]
                self.G.eval()
                # Changing hair color, gender, and age
                for j in range(self.c_dim):
                    # fake_c = torch.cat([fixed_c1_list[j], fixed_zero1, fixed_mask1], dim=1)
                    fake_image_list.append(self.G(fixed_x2, fixed_c1_list[j], branch_id=1))
                # Changing emotional expressions
                for j in range(self.c2_dim):
                    # fake_c = torch.cat([fixed_zero2, fixed_c2_list[j], fixed_mask2], dim=1)
                    fake_image_list.append(self.G(fixed_x2, fixed_c2_list[j], branch_id=2))
                fake = torch.cat(fake_image_list, dim=3)

                self.G.train()
                # Save the translated images
                save_image(self.denorm(fake.data),
                           os.path.join(self.sample_path, '{}_fake_x2.png'.format(i + 1)), nrow=1, padding=0)


            # Save model checkpoints
            if (i+1) % self.model_save_step == 0 and (i+1)>=self.model_save_star:
                torch.save(self.G.state_dict(),
                    os.path.join(self.model_save_path, '{}_G.pth'.format(i+1)))
                torch.save(self.D.state_dict(),
                    os.path.join(self.model_save_path, '{}_D.pth'.format(i+1)))

            # Decay learning rate
            decay_step = 1000
            if (i+1) > (self.num_iters - self.num_iters_decay) and (i+1) % decay_step==0:
                g_lr -= (self.g_lr / float(self.num_iters_decay) * decay_step)
                d_lr -= (self.d_lr / float(self.num_iters_decay) * decay_step)
                self.update_lr(g_lr, d_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

            if (i+1) == self.stage1:
                return

    def test(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()

        if self.dataset == 'CelebA':
            data_loader = self.dataset1_loader
        else:
            data_loader = self.dataset2_loader

        for i, (real_x, org_c) in enumerate(data_loader):
            real_x = self.to_var(real_x, volatile=True)

            if self.dataset == 'CelebA':
                target_c_list = self.make_celeb_labels(org_c)
            else:
                target_c_list = []
                for j in range(self.c_dim):
                    target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
                    target_c_list.append(self.to_var(target_c, volatile=True))

            # Start translations
            fake_image_list = [real_x]
            for target_c in target_c_list:
                fake_image_list.append(self.G(real_x, target_c))
            fake_images = torch.cat(fake_image_list, dim=3)
            save_path = os.path.join(self.result_path, '{}_fake.png'.format(i+1))
            save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
            print('Translated test images and saved into "{}"..!'.format(save_path))

    def test_multi(self):
        """Facial attribute transfer and expression synthesis on CelebA."""
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()
        loader2 = iter(self.dataset2_loader)
        for i, (real_x, org_c) in enumerate(self.dataset1_loader):
            real_x, _ = next(loader2)
            # Prepare input images and target domain labels
            real_x = self.to_var(real_x, volatile=True)
            target_c1_list = self.make_celeb_labels(org_c)
            target_c2_list = []
            for j in range(self.c2_dim):
                target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c2_dim)
                target_c2_list.append(self.to_var(target_c, volatile=True))

            # Zero vectors and mask vectors
            zero1 = self.to_var(torch.zeros(real_x.size(0), self.c2_dim))     # zero vector for rafd expressions
            mask1 = self.to_var(self.one_hot(torch.zeros(real_x.size(0)), 2)) # mask vector: [1, 0]
            zero2 = self.to_var(torch.zeros(real_x.size(0), self.c_dim))      # zero vector for celebA attributes
            mask2 = self.to_var(self.one_hot(torch.ones(real_x.size(0)), 2))  # mask vector: [0, 1]

            # Changing hair color, gender, and age
            fake_image_list = [real_x]
            for j in range(self.c_dim):
                target_c = torch.cat([target_c1_list[j], zero1, mask1], dim=1)
                fake_image_list.append(self.G(real_x, target_c))

            # Changing emotional expressions
            for j in range(self.c2_dim):
                target_c = torch.cat([zero2, target_c2_list[j], mask2], dim=1)
                fake_image_list.append(self.G(real_x, target_c))
            fake_images = torch.cat(fake_image_list, dim=3)

            # Save the translated images
            save_path = os.path.join(self.result_path, '{}_fake.png'.format(i+1))
            save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
            print('Translated test images and saved into "{}"..!'.format(save_path))

    def test_multi_branch(self):
        """Facial attribute transfer and expression synthesis on CelebA."""
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()
        loader2 = iter(self.dataset2_loader)
        for i, (real_x, org_c) in enumerate(self.dataset1_loader):
            real_x, _ = next(loader2)
            # Prepare input images and target domain labels
            real_x = self.to_var(real_x, volatile=True)
            target_c1_list = self.make_celeb_labels(org_c)
            target_c2_list = []
            for j in range(self.c2_dim):
                target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c2_dim)
                target_c2_list.append(self.to_var(target_c, volatile=True))

            # Zero vectors and mask vectors
            zero1 = self.to_var(torch.zeros(real_x.size(0), self.c2_dim))     # zero vector for rafd expressions
            mask1 = self.to_var(self.one_hot(torch.zeros(real_x.size(0)), 2)) # mask vector: [1, 0]
            zero2 = self.to_var(torch.zeros(real_x.size(0), self.c_dim))      # zero vector for celebA attributes
            mask2 = self.to_var(self.one_hot(torch.ones(real_x.size(0)), 2))  # mask vector: [0, 1]

            # Changing hair color, gender, and age
            fake_image_list = [real_x]
            for j in range(self.c_dim):
                # target_c = torch.cat([target_c1_list[j], zero1, mask1], dim=1)
                fake_image_list.append(self.G(real_x, target_c1_list[j], branch_id=1))

            # Changing emotional expressions
            for j in range(self.c2_dim):
                # target_c = torch.cat([zero2, target_c2_list[j], mask2], dim=1)
                fake_image_list.append(self.G(real_x, target_c2_list[j], branch_id=2))
            fake_images = torch.cat(fake_image_list, dim=3)

            # Save the translated images
            save_path = os.path.join(self.result_path, '{}_fake.png'.format(i+1))
            save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
            print('Translated test images and saved into "{}"..!'.format(save_path))

    def demo(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()

        if self.dataset == 'CelebA':
            data_loader = self.dataset1_loader
        else:
            data_loader = self.dataset2_loader

        for i, real_x in enumerate(data_loader):
            real_x = self.to_var(real_x, volatile=True)
            print(real_x.size())
            if self.dataset == 'CelebA':
                target_c_list = self.make_celeb_labels(torch.FloatTensor([[1, 0, 0, 0, 1]]))
            else:
                target_c_list = []
                for j in range(self.c_dim):
                    target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
                    target_c_list.append(self.to_var(target_c, volatile=True))

            # Start translations
            fake_image_list = [real_x]
            for target_c in target_c_list:
                fake_image_list.append(self.G(real_x, target_c))
            fake_images = torch.cat(fake_image_list, dim=3)
            save_path = os.path.join('demo', '{}_fake.png'.format(i+1))
            save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
            print('Translated test images and saved into "{}"..!'.format(save_path))

    def evaluate(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        for i in range(self.test_interval[0], self.test_interval[1]+1, self.test_step):
            G_model_name = '{}_{}_G'.format(i, self.test_iters)
            print('Test model {} ...'.format(G_model_name))
            G_path = os.path.join(self.model_save_path, G_model_name + '.pth')
            self.G.load_state_dict(torch.load(G_path))
            self.G.eval()

            data_loader = self.dataset2_loader
            classes = data_loader.dataset.classes
            if not os.path.exists(self.test_save_path):
                os.makedirs(self.test_save_path)
            if not os.path.exists(os.path.join(self.test_save_path, G_model_name)):
                os.makedirs(os.path.join(self.test_save_path, G_model_name))
            for Class in data_loader.dataset.classes:
                if not os.path.exists(os.path.join(self.test_save_path, G_model_name, Class)):
                    os.makedirs(os.path.join(self.test_save_path, G_model_name, Class))

            for i, (real_x, real_label) in enumerate(data_loader):
                real_x = self.to_var(real_x, volatile=True)
                real_c = self.one_hot(real_label, self.c_dim)
                real_c = self.to_var(real_c, volatile=True)


                target_c_list = []
                if self.test_traverse:
                    for j in range(self.c_dim):
                        target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
                        target_c_list.append(self.to_var(target_c, volatile=True))

                    # Start translations
                    for j, target_c in enumerate(target_c_list):
                        fake = self.G(real_x, target_c)[0].detach()
                        save_path = os.path.join(self.test_save_path, G_model_name, classes[j], '{}_{}.png'.format(i, j))
                        save_image(self.denorm(fake.data), save_path)
                        print('Translated test images and saved into "{}"..!'.format(save_path))

                else:
                    target_c_list = [real_c]

                    # Start translations
                    for j, target_c in enumerate(target_c_list):
                        fake = self.G(real_x, target_c)[0].detach()
                        save_path = os.path.join(self.test_save_path, G_model_name, classes[real_label[0]], '{}_{}.png'.format(i, classes[real_label[0]]))
                        save_image(self.denorm(fake.data), save_path)
                        print('Translated test images and saved into "{}"..!'.format(save_path))

    def evaluate_multi(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        for i in range(self.test_interval[0], self.test_interval[1]+1, self.test_step):
            G_model_name = '{}_G'.format(i)
            print('Test model {} ...'.format(G_model_name))
            G_path = os.path.join(self.model_save_path, G_model_name + '.pth')
            self.G.load_state_dict(torch.load(G_path))
            self.G.eval()

            data_loader = self.dataset2_loader
            classes = data_loader.dataset.classes
            if not os.path.exists(self.test_save_path):
                os.makedirs(self.test_save_path)
            if not os.path.exists(os.path.join(self.test_save_path, G_model_name)):
                os.makedirs(os.path.join(self.test_save_path, G_model_name))
            for Class in data_loader.dataset.classes:
                if not os.path.exists(os.path.join(self.test_save_path, G_model_name, Class)):
                    os.makedirs(os.path.join(self.test_save_path, G_model_name, Class))

            for i, (real_x, real_label) in enumerate(data_loader):
                real_x = self.to_var(real_x, volatile=True)
                real_c = self.one_hot(real_label, self.c_dim)
                real_c = self.to_var(real_c, volatile=True)

                zero2 = self.to_var(torch.zeros(real_x.size(0), self.c_dim))  # zero vector for celebA attributes
                mask2 = self.to_var(self.one_hot(torch.ones(1), 2))  # mask vector: [0, 1]

                target_c_list = []
                if self.test_traverse:
                    for j in range(self.c_dim):
                        target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
                        target_c_list.append(self.to_var(target_c, volatile=True))

                    # Start translations
                    for j, target_c in enumerate(target_c_list):
                        target_c = torch.cat([zero2, target_c, mask2], dim=1)
                        fake = self.G(real_x, target_c)[0].detach()
                        save_path = os.path.join(self.test_save_path, G_model_name, classes[j], '{}_{}.png'.format(i, j))
                        save_image(self.denorm(fake.data), save_path)
                        print('Translated test images and saved into "{}"..!'.format(save_path))

                else:
                    target_c_list = [real_c]

                    # Start translations
                    for j, target_c in enumerate(target_c_list):
                        target_c = torch.cat([zero2, target_c, mask2], dim=1)
                        fake = self.G(real_x, target_c)[0].detach()
                        save_path = os.path.join(self.test_save_path, G_model_name, classes[real_label[0]], '{}_{}.png'.format(i, classes[real_label[0]]))
                        save_image(self.denorm(fake.data), save_path)
                        print('Translated test images and saved into "{}"..!'.format(save_path))

    def vis(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_{}_G.pth'.format(self.test_interval[0], self.test_iters))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()
        vis = visdom.Visdom()

        if self.dataset == 'CelebA':
            data_loader = self.dataset1_loader
        else:
            data_loader = self.dataset2_loader

        for i, (real_x, real_label) in enumerate(data_loader):
            real_x = self.to_var(real_x, volatile=True)
            real_c = self.one_hot(real_label, self.c_dim)
            real_c = self.to_var(real_c, volatile=True)
            if i < 300:
                continue
            target_c_list = []
            for j in range(self.c_dim):
                target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
                target_c_list.append(self.to_var(target_c, volatile=True))

            # Start translations
            # image
            # for j, target_c in enumerate(target_c_list):
            #     m, b, f, im = self.G(real_x, target_c, branch_id=2, vis_flag=True)
            #     if j == 0:
            #         vis.image((real_x.cpu().data[0]+1)*127.5)
            #     vis.image((im.cpu().data[0]+1)*127.5)
            # for j, target_c in enumerate(target_c_list):
            #     m, b, f, im = self.G(real_x, target_c, branch_id=2, vis_flag=True)

            #     # if j == 0:
            #     #     vis.image((real_x.cpu().data[0]+1)*127.5)
            #     # vis.image((im.cpu().data[0]+1)*127.5)
            #     res = np.abs(im.cpu().data[0] - real_x.cpu().data[0])
            #     res_ = res.numpy().max(axis=0)
            #     res_ = np.flip(res_, 0)
            #     vis.heatmap(res_)
            for j, target_c in enumerate(target_c_list):
                m, b, f, im, im_b, im_mf, im_f, im_m, im_fb = self.G(real_x, target_c, branch_id=2, vis_flag=True)
                # m, f, im, im_m, im_f = self.G(real_x, target_c, branch_id=2, vis_flag=True)

                # if j == 0:
                #     vis.image((real_x.cpu().data[0]+1)*127.5)
                vis.image((im.cpu().data[0]+1)*127.5)
                vis.image((im_b.cpu().data[0]+1)*127.5)
                vis.image((im_mf.cpu().data[0]+1)*127.5)
                vis.image((im_f.cpu().data[0]+1)*127.5)
                vis.image((im_m.cpu().data[0]+1)*127.5)
                for w in range(1, 6):
                    im = self.G.act(self.G.smooth(0.2*w*m*f))
                    vis.image((im.cpu().data[0]+1)*127.5)
                # vis.image((im_fb.cpu().data[0]+1)*127.5)
                # res = np.abs(im.cpu().data[0] - real_x.cpu().data[0])
                # res_ = res.numpy().max(axis=0)
                # res_ = np.flip(res_, 0)
                # vis.heatmap(res_)
                # m_ = (m).cpu().data.numpy()[0]
                # m_ = m_.max(axis=0)
                # m_ = np.flip(m_, 0)
                # # vis.heatmap(m_)
                # vis.image(((real_x+1)*127.5*m[:,0:3]).cpu().data[0])
                # vis.image((im.cpu().data[0]+1)*127.5)
                # m__ = (m).cpu().data.numpy()[0]
                # m__ = m__[0]
                # m__ = np.flip(m__, 0)
                # vis.heatmap(m_-m__)
                # f_ = (f).cpu().data.numpy()[0]
                # f_ = f_.max(axis=0)
                # f_ = np.flip(f_, 0)
                # vis.heatmap(f_)
                # mf_ = (m*f).cpu().data.numpy()[0]
                # mf_ = mf_.max(axis=0)
                # mf_ = np.flip(mf_, 0)
                # vis.heatmap(mf_)
                # b_ = (b).cpu().data.numpy()[0]
                # b_ = b_.max(axis=0)
                # b_ = np.flip(b_, 0)
                # vis.heatmap(b_)
                # mfb_ = (m*f+b).cpu().data.numpy()[0]
                # mfb_ = mfb_.max(axis=0)
                # mfb_ = np.flip(mfb_, 0)
                # vis.heatmap(mfb_)
                # print('m:{} f:{} mf:{}, b:{}'.format(m_.max(), f_.max(), mf_.max(), b_.max()))
                # for channel in range(0, 64):
                #     # vis.text('m channel {}'.format(channel))
                #     x = m.cpu().data.numpy()[0][channel]
                #     x = np.rot90(x)
                #     x = np.rot90(x)
                #     vis.heatmap(x)
                #     y = f.cpu().data.numpy()[0][channel]
                #     y = np.rot90(y)
                #     y = np.rot90(y)
                #     vis.heatmap(y)
                #     vis.heatmap(x*y)
                #     z = b.cpu().data.numpy()[0][channel]
                #     z = np.rot90(z)
                #     z = np.rot90(z)
                #     vis.heatmap(z)
                # break
            if i == 5:
                break