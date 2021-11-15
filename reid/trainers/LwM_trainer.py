# -*- coding: utf-8 -*-
# Time    : 2021/8/9 15:22
# Author  : Yichen Lu


from reid.trainers import IncrementalTrainer
from reid.loss import distillation


class LwMTrainer(IncrementalTrainer):
    def __init__(self,
                 networks_prec,
                 networks,
                 incremental_classes,
                 optimizer,
                 lr_scheduler
                 ):
        super(LwMTrainer, self).__init__(networks_prec,
                                         networks,
                                         incremental_classes,
                                         optimizer,
                                         lr_scheduler,
                                         )

        self.class_criterion = distillation.ClassificationDistillationLoss(T=2.)
        self.attention_criterion = distillation.GradCAMLoss()
        self.register_GradCAM_hook()

    def register_GradCAM_hook(self):
        forward_hook_t = self.networks_prec.backbone.create_forward_hook('activation')
        self.networks_prec.backbone.forward_hooks.append(self.networks_prec.backbone.base.layer4[-1].conv3.register_forward_hook(forward_hook_t))
        backward_hook_t = self.networks_prec.backbone.create_backward_hook('grads')
        self.networks_prec.backbone.backward_hooks.append(self.networks_prec.backbone.base.layer4[-1].conv3.register_backward_hook(backward_hook_t))

        forward_hook_s = self.networks_prec.backbone.create_forward_hook('activation')
        self.networks_prec.backbone.forward_hooks.append(self.networks_prec.backbone.base.layer4[-1].conv3.register_forward_hook(forward_hook_s))
        backward_hook_s = self.networks_prec.backbone.create_backward_hook('grads')
        self.networks_prec.backbone.backward_hooks.append(self.networks_prec.backbone.base.layer4[-1].conv3.register_backward_hook(backward_hook_s))

    def GradCAM(self, preds, preds_old, pids):
        self.networks_prec.zero_grad()
        self.networks.zero_grad()

        top_one = preds.argmax(dim=1, keepdim=True)
        top_logits = preds.gather(dim=1, index=top_one)
        top_logits.mean().backward(retain_graph=True)

        top_logits_old = preds_old.gather(dim=1, index=top_one)
        top_logits_old.mean().backward()

        self.networks_prec.zero_grad()
        self.networks.zero_grad()

        return [self.networks.backbone.forward_records['activation'],
                self.networks.backbone.backward_records['grads'],
                self.networks_prec.backbone.forward_records['activation'].detach(),
                self.networks_prec.backbone.backward_records['grads'].detach()]

    def _compute_loss(self, outputs, outputs_prec, pids):
        pooled, preds = outputs["global"], outputs["preds"]
        pid_loss, triplet_loss = self.basic_criterion(pooled, preds, pids)

        distillation_loss = self.class_criterion(preds[..., self.incremental_classes:], outputs_prec["preds"])

        act_s, grads_s, act_t, grads_t = self.GradCAM(preds[..., self.incremental_classes:], outputs_prec["preds"], pids)
        gradCAM_loss = self.attention_criterion(act_s, grads_s, act_t.cuda(0), grads_t.cuda(0))
        distillation_loss = distillation_loss + gradCAM_loss

        loss = pid_loss + triplet_loss + distillation_loss
        return loss, [pid_loss.item(), triplet_loss.item(), distillation_loss.item()]
