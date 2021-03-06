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
from source.dis_model.discriminator import *
from source.gen_model.generator import *
from updater import *
from PIL import Image
from source.evaluation.fnd import *
import collections


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('InstanceNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Solver(object):

    def __init__(self, dataset1_loader, dataset2_loader, eval_loader, config):
        self.device_ids = config.device_ids

        self.debug = config.debug

        # Data loader
        self.dataset1_loader = dataset1_loader
        self.dataset2_loader = dataset2_loader
        self.eval_loader = eval_loader

        # Model hyper-parameters
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.d2_repeat_num = config.d2_repeat_num
        self.n_hiddens = config.n_hiddens
        self.d_train_repeat = config.d_train_repeat


        # Hyper-parameteres
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_reg = config.lambda_reg
        self.lambda_gp  = config.lambda_gp
        self.lambda_kl =config.lambda_kl

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

        # Evaluation settings
        self.eval_batchsize = config.eval_batchsize
        self.dims = config.dims
        self.eval_model = config.eval_model
        self.eval_path = self.eval_loader.dataset.root

        # Test settings
        self.test_model = config.test_model
        self.test_interval = config.test_interval
        self.test_iters = config.test_iters
        self.test_step = config.test_step
        self.test_traverse = config.test_traverse
        self.test_save_path = os.path.join(config.ckpt_path, config.task_name, config.evaluates)

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
        # self.G = Generator(conv_dim=self.g_conv_dim, c_dim=self.c_dim, repeat_num=self.g_repeat_num)
        # self.G = MultiViewGenerator(conv_dim=self.g_conv_dim, c_dim=self.c_dim, repeat_num=self.g_repeat_num)
        # self.G = MultiSampleGenerator(conv_dim=self.g_conv_dim, c_dim=self.c_dim, repeat_num=self.g_repeat_num)
        # self.G = CrossAttentionMultiSampleGenerator(conv_dim=self.g_conv_dim, c_dim=self.c_dim, repeat_num=self.g_repeat_num)
        # self.G = CINSelfCrossAttentionMultiSampleGenerator(ch=self.g_conv_dim, n_classes=self.c_dim, n_repeat=self.g_repeat_num)
        # self.G = CINCrossAttentionMultiSampleGenerator(ch=self.g_conv_dim, n_classes=self.c_dim, n_repeat=self.g_repeat_num)
        self.G = CINCrossAttentionMultiSampleFullResidualGenerator(ch=self.g_conv_dim, n_classes=self.c_dim, n_repeat=self.g_repeat_num)
        # self.G = CINGenerator(ch=self.g_conv_dim, n_classes=self.c_dim, n_repeat=self.g_repeat_num)
        # self.D = Discriminator(image_size=self.image_size, conv_dim=self.d_conv_dim, c_dim=self.c_dim, repeat_num=self.d_repeat_num)
        self.D = ProjectionDiscriminator(image_size=self.image_size, conv_dim=self.d_conv_dim, c_dim=self.c_dim,
                               repeat_num=self.d_repeat_num)
        # self.G.apply(weights_init_normal)
        # self.D.apply(weights_init_normal)
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
        # todo
        if self.dataset2_loader is not None:
            self.D2 = VariableDiscriminator(conv_dim=self.d_conv_dim, c_dim=self.c_dim, repeat_num=self.d2_repeat_num,
                                            n_hiddens=self.n_hiddens)
            # self.D2 = PairwiseDiscriminator(conv_dim=self.d_conv_dim, c_dim=self.c_dim, repeat_num=self.d2_repeat_num,
            #                                 n_hiddens=self.n_hiddens)
            # self.D2 = torch.nn.DataParallel(self.D2, device_ids=self.device_ids)
            self.d2_optimizer = torch.optim.Adam(self.D2.parameters(), self.d_lr, [self.beta1, self.beta2])
            self.print_network(self.D2, 'D2')
            if torch.cuda.is_available():
                self.D2.cuda()

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
        # todo
        if self.dataset2_loader is not None:
            self.D2.load_state_dict(torch.load(os.path.join(
                self.model_save_path, '{}_D2.pth'.format(self.resume))))
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
        # todo
        if self.dataset2_loader is not None:
            self.d2_optimizer.zero_grad()

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
        # todo:
        if x.dim() > 1:
            _, predicted = torch.max(x, dim=1)
            correct = (predicted == y).float()
        else:
            n_intervals = n_classes - 1
            eps = 1. / n_classes / 2.
            target = y.float()/(n_intervals*0.5)-1.
            correct = ( torch.abs(x - target) < eps ).float()
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
            fixed_c = self.one_hot(torch.ones(fixed_x.size(0)) * i, self.c_dim)
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
                real_label = self.to_var(real_label, requires_grad=False)   # this is same as real_c if dataset == 'CelebA'
                fake_label = self.to_var(fake_label, requires_grad=False)
                
                # ================== Train D ================== #

                # Compute loss with real images
                out_real, out_cls, out_reg = self.D(real_x, real_label)
                # Compute loss with fake images
                fake_x = self.G(real_x, fake_c).detach()
                out_fake, _, _ = self.D(fake_x, fake_label)

                # d_loss_adv = loss_hinge_dis(out_fake, out_real)
                d_loss_adv = -torch.mean(out_real) + torch.mean(out_fake)
                d_loss_cls = F.cross_entropy(out_cls, real_label)
                # todo:regression
                d_loss_reg = loss_hard_reg(out_reg, real_label, self.c_dim)

                # Compute classification accuracy of the discriminator
                if (i+1) % self.log_step == 0:
                    classification_accuracies = self.compute_accuracy(out_cls, real_label, n_classes=self.c_dim)
                    regression_accuracies = self.compute_accuracy(out_reg.squeeze(), real_label, n_classes=self.c_dim)

                    log = "{:.2f}/{:.2f}".format(classification_accuracies.data.cpu().numpy(), regression_accuracies.data.cpu().numpy())
                    print('Classification/regression Acc: ', end='')
                    print(log)


                # Backward + Optimize
                d_loss = d_loss_adv + self.lambda_cls * d_loss_cls + self.lambda_reg * d_loss_reg
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Compute gradient penalty
                alpha = torch.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real_x)
                interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
                out, _, _ = self.D(interpolated, real_label)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()


                # Logging
                loss = collections.OrderedDict()
                loss['D/loss_adv'] = d_loss_adv.data
                loss['D/loss_reg'] = d_loss_reg.data
                loss['D/loss_gp'] = d_loss_gp.data
                loss['D/loss_cls'] = d_loss_cls.data

                # todo
                if self.dataset2_loader is not None:
                    pExample1_img, pExample2_img, pExample1_lbl, pExample2_lbl, nExample1_img, nExample2_img, nExample1_lbl, nExample2_lbl = iter(
                        self.dataset2_loader).next()

                    # Generat fake labels randomly (target domain labels)

                    pExample1_c = self.one_hot(pExample1_lbl, self.c_dim)
                    pExample2_c = self.one_hot(pExample2_lbl, self.c_dim)
                    nExample1_c = self.one_hot(nExample1_lbl, self.c_dim)
                    nExample2_c = self.one_hot(nExample2_lbl, self.c_dim)

                    # Convert tensor to variable
                    pExample1_img = self.to_var(pExample1_img)
                    pExample2_img = self.to_var(pExample2_img)
                    nExample1_img = self.to_var(nExample1_img)
                    nExample2_img = self.to_var(nExample2_img)

                    pExample1_c = self.to_var(pExample1_c, self.c_dim)
                    pExample2_c = self.to_var(pExample2_c, self.c_dim)
                    nExample1_c = self.to_var(nExample1_c, self.c_dim)
                    nExample2_c = self.to_var(nExample2_c, self.c_dim)

                    pExample1_lbl = self.to_var(pExample1_lbl, requires_grad=False)
                    pExample2_lbl = self.to_var(pExample2_lbl, requires_grad=False)
                    nExample1_lbl = self.to_var(nExample1_lbl, requires_grad=False)
                    nExample2_lbl = self.to_var(nExample2_lbl, requires_grad=False)

                    # ================== Train D2 ================== #

                    # Compute loss with real example
                    out_real, mu1_real, logvar1_real, mu2_real, logvar2_real = self.D2(pExample1_img, pExample2_img, pExample1_lbl,
                                                                   pExample2_lbl)
                    # Compute loss with negtive example
                    out_neg, mu1_neg, logvar1_neg, mu2_neg, logvar2_neg = self.D2(nExample1_img, nExample2_img, nExample1_lbl,
                                                                    nExample2_lbl)
                    # no projection
                    # out_neg, mu1_neg, logvar1_neg, mu2_neg, logvar2_neg = self.D2(nExample1_img, nExample2_img)

                    # # Compute loss with real example
                    # out_real = self.D2(pExample1_img, pExample2_img, pExample1_lbl, pExample2_lbl)
                    # # Compute loss with negtive example
                    # out_neg = self.D2(nExample1_img, nExample2_img, nExample1_lbl, nExample2_lbl)

                    # Compute loss with fake example
                    # fExample2_img = self.G(pExample1_img, pExample2_c).detach()
                    # out_fake, _, _, mu2_fake, logvar2_fake = self.D2(pExample1_img, fExample2_img, pExample1_lbl, pExample2_lbl)

                    fExample2_img = self.G(pExample1_img, pExample2_c).detach()
                    out_fake, _, _, mu2_fake, logvar2_fake = self.D2(pExample1_img, fExample2_img, pExample1_lbl,
                                                                     pExample2_lbl)

                    # unilateral projection
                    # out_fake, _, _, mu2_fake, logvar2_fake = self.D2(pExample1_img, fExample2_img, None,
                    #                                                  pExample2_lbl)

                    kl_real = loss_kl(mu1_real, logvar1_real) + loss_kl(mu2_real, logvar2_real)
                    kl_neg = loss_kl(mu1_neg, logvar1_neg) + loss_kl(mu2_neg, logvar2_neg)
                    kl_fake = loss_kl(mu2_fake, logvar2_fake)

                    kl = (kl_fake + kl_neg + kl_real) * 0.2
                    # d_loss_adv = loss_hinge_dis(out_fake, out_real)
                    d2_loss_adv = -torch.mean(out_real) + (torch.mean(out_fake) + torch.mean(out_neg))*0.5

                    d2_loss = d2_loss_adv + kl*self.lambda_kl

                    # Backward + Optimize
                    self.reset_grad()
                    d2_loss.backward()
                    self.d2_optimizer.step()

                    # Compute gradient penalty
                    alpha = torch.rand(nExample2_img.size(0), 1, 1, 1).cuda().expand_as(nExample2_img)
                    interpolated = Variable(alpha * nExample2_img.data + (1 - alpha) * fExample2_img.data, requires_grad=True)
                    out, _, _, _, _ = self.D2(pExample1_img, interpolated, pExample1_lbl, pExample2_lbl)

                    grad = torch.autograd.grad(outputs=out,
                                               inputs=interpolated,
                                               grad_outputs=torch.ones(out.size()).cuda(),
                                               retain_graph=True,
                                               create_graph=True,
                                               only_inputs=True)[0]

                    grad = grad.view(grad.size(0), -1)
                    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                    d2_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                    # Backward + Optimize
                    d2_loss = self.lambda_gp * d2_loss_gp
                    self.reset_grad()
                    d2_loss.backward()
                    self.d2_optimizer.step()

                    # Logging
                    loss['D2/loss_adv'] = d2_loss_adv.data
                    loss['D2/d2_loss_gp'] = d2_loss_gp.data
                    loss['D2/loss_kl'] = kl.data

                # ================== Train G ================== #
                if (i+1) % self.d_train_repeat == 0:

                    # Original-to-target and target-to-original domain
                    fake_x = self.G(real_x, fake_c)
                    # todo
                    rec_x  = self.G(real_x, real_c)

                    # Compute losses
                    out_fake, out_cls, out_reg = self.D(fake_x, fake_label)
                    g_loss_adv = -torch.mean(out_fake)
                    g_loss_rec = torch.mean(torch.abs(real_x - rec_x))
                    g_loss_cls = F.cross_entropy(out_cls, fake_label)
                    g_loss_reg = loss_hard_reg(out_reg, fake_label, self.c_dim)

                    # todo
                    if self.dataset2_loader is not None:
                        # fExample2_img = self.G(pExample1_img, pExample2_c)
                        # out_fake, _, _, _, _ = self.D2(pExample1_img, fExample2_img,
                        #                                pExample1_lbl,
                        #                                pExample2_lbl)
                        fExample2_img = self.G(pExample1_img, nExample2_c)
                        out_fake, _, _, _, _ = self.D2(pExample1_img, fExample2_img,
                                                       pExample1_lbl,
                                                       nExample2_lbl)
                        # unilateral projection
                        # out_fake, _, _, _, _ = self.D2(pExample1_img, fExample2_img,
                        #                                None,
                        #                                nExample2_lbl)
                        g_loss_adv2 = -torch.mean(out_fake)
                    else:
                        g_loss_adv2 = 0

                    # Backward + Optimize
                    g_loss = g_loss_adv + self.lambda_cls * g_loss_cls + self.lambda_rec * g_loss_rec + self.lambda_reg * g_loss_reg + g_loss_adv2
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    loss['G/loss_adv'] = g_loss_adv.data
                    loss['G/loss_rec'] = g_loss_rec.data
                    loss['G/loss_reg'] = g_loss_reg.data
                    loss['G/loss_cls'] = g_loss_cls.data
                    loss['G/loss_adv2'] = g_loss_adv2



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
                    if self.dataset2_loader is not None:
                        pair_images = torch.cat([pExample1_img, pExample2_img, fExample2_img], dim=3).detach()
                        if self.debug:
                            pair_images = torch.cat([pair_images, nExample1_img, nExample2_img], dim=3).detach()
                        save_image(self.denorm(pair_images.data),
                                   os.path.join(self.sample_path, '{}_{}_pair_images.png'.format(e + 1, i + 1)), nrow=1, padding=0)

                    print('Translated images and saved into {}..!'.format(self.sample_path))

            # Save model checkpoints
            if (e+1) % self.model_save_step == 0 and (e+1) > self.model_save_star:
                print('Save model checkpoints')
                torch.save(self.G.state_dict(),
                    os.path.join(self.model_save_path, '{}_G.pth'.format(e+1)))
                torch.save(self.D.state_dict(),
                    os.path.join(self.model_save_path, '{}_D.pth'.format(e+1)))
                if self.dataset2_loader is not None:
                    torch.save(self.D2.state_dict(),
                               os.path.join(self.model_save_path, '{}_D2.pth'.format(e + 1)))

                intra_fid = calculate_intra_fid(self.eval_path, self.eval_batchsize, True, self.dims, self.eval_model,
                                                self.G, self.eval_loader)
                log = 'TEST Epoch [{}/{}]'.format(e+1, self.num_epochs)
                for tag, value in intra_fid.items():
                    log += ", {}: {:.4f}".format(tag, value)
                test_log_path = os.path.join(self.log_path, 'test.log')
                with open(test_log_path, 'a') as f:
                    f.write(log)
                    f.write('\n')
                print(log)

            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()

        data_loader = self.dataset1_loader

        for i, (real_x, org_c) in enumerate(data_loader):
            real_x = self.to_var(real_x, requires_grad=False)
            target_c_list = []
            for j in range(self.c_dim):
                target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
                target_c_list.append(self.to_var(target_c, requires_grad=False))

            # Start translations
            fake_image_list = [real_x]
            for target_c in target_c_list:
                fake_image_list.append(self.G(real_x, target_c))
            fake_images = torch.cat(fake_image_list, dim=3)
            save_path = os.path.join(self.result_path, '{}_fake.png'.format(i+1))
            save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
            print('Translated test images and saved into "{}"..!'.format(save_path))


    def evaluate(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        for i in range(self.test_interval[0], self.test_interval[1]+1, self.test_step):
            G_model_name = '{}_G'.format(i)
            print('Test model {} ...'.format(G_model_name))
            G_path = os.path.join(self.model_save_path, G_model_name + '.pth')
            self.G.load_state_dict(torch.load(G_path))
            self.G.eval()

            data_loader = self.dataset1_loader
            classes = data_loader.dataset.classes
            if not os.path.exists(self.test_save_path):
                os.makedirs(os.path.join(self.test_save_path))
            if not os.path.exists(os.path.join(self.test_save_path, G_model_name)):
                os.makedirs(os.path.join(self.test_save_path, G_model_name))
            for Class in data_loader.dataset.classes:
                if not os.path.exists(os.path.join(self.test_save_path, G_model_name, Class)):
                    os.makedirs(os.path.join(self.test_save_path, G_model_name, Class))

            for i, (real_x, real_label) in enumerate(data_loader):
                real_x = self.to_var(real_x, requires_grad=False)
                real_c = self.one_hot(real_label, self.c_dim)
                real_c = self.to_var(real_c, requires_grad=False)


                target_c_list = []
                for j in range(self.c_dim):
                    target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
                    target_c_list.append(self.to_var(target_c, requires_grad=False))

                for j, target_c in enumerate(target_c_list):
                    fake = self.G(real_x, target_c)[0].detach()
                    save_path = os.path.join(self.test_save_path, G_model_name, classes[j], '{}_{}.png'.format(i, j))
                    save_image(self.denorm(fake.data), save_path, padding=0)
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