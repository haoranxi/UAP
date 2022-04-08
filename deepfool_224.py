# coding: utf-8
import numpy as np
import torch as torch
#from torch.autograd.gradcheck import zero_gradients


def deepfool_target(image, net, device, num_classes = 1000, target_class=2, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param device: device to use
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    
    image = image.to(device)
    net = net.to(device)

    net.eval() 
    f_image = net.forward(image.unsqueeze(0)).detach().cpu().numpy().flatten()

    net.train() 
    
    I = (np.array(f_image)).flatten().argsort()[::-1]
    
    #I = (np.array(f_image)).flatten()

    I = I[0:num_classes]
    #index = np.argmax(f_image)
    #print(index)
    label = I[0]

    input_shape = image.shape
    pert_image = image

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = pert_image.unsqueeze(0)
    x.requires_grad_()

    net.eval() 
    fs = net.forward(x)
    net.train() 
    
    fs_list = [fs[0,I[k]] for k in range(num_classes)]

    k_i = label
    # and label != target_class

    #while k_i == label and loop_i < max_iter:
    while k_i != target_class and loop_i < max_iter:

        pert = np.inf
        sss = fs[0, I[0]]
        sss.backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for kkk in range(1, 2):
            k = target_class

            #zero_gradients(x)
            if x.grad is not None:
                x.grad.zero_()
            #x.zero_grad()
            
            fff = fs[0, k]

            fff.backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            #w_k = - cur_grad + grad_orig
            #f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()
            f_k = (fs[0, k] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        r_i = (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).to(device)
        #pert_image = image - (1+overshoot)*torch.from_numpy(r_tot).to(device)

        x = pert_image
        x.requires_grad_()
        
        net.eval() 
        fs = net.forward(x)
        net.train() 
        
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1
    '''
    if k_i == label:
        print("bad")
        print(k_i)
    else:
        print("good")
        print(k_i)
        print(loop_i)
    '''
    r_tot = (1+overshoot)*r_tot

    if k_i != target_class:
        temp = 0
    else:
        temp = 1


    return r_tot, loop_i, label, k_i, pert_image,temp






def deepfool(image, net, device, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param device: device to use
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    
    image = image.to(device)
    net = net.to(device)

    net.eval() 
    f_image = net.forward(image.unsqueeze(0)).detach().cpu().numpy().flatten()

    net.train() 
    
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.shape
    pert_image = image

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = pert_image.unsqueeze(0)
    x.requires_grad_()

    net.eval() 
    fs = net.forward(x)
    net.train() 
    
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            #zero_gradients(x)
            if x.grad is not None:
                x.grad.zero_()
            #x.zero_grad()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        r_i = (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).to(device)

        x = pert_image
        x.requires_grad_()
        
        net.eval() 
        fs = net.forward(x)
        net.train() 
        
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    if k_i != target_class:
        r_tot = np.zeros(input_shape)


    return r_tot, loop_i, label, k_i, pert_image
