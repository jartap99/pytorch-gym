#
# @rajp
# 

from utils import * 
import copy
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

torch.manual_seed(876)

""" post training static quantization
    https://pytorch.org/blog/introduction-to-quantization-on-pytorch/ 
    https://pytorch.org/docs/stable/quantization.html
    https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html 

"""
    
from ptsq import LeNet5
from ptsq import step1
                     
def step2(model_fp32):
    optimizer = torch.optim.SGD(model_fp32.parameters(), lr=0.0001)
    model_fp32.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    
    qmodel = torch.ao.quantization.prepare_qat(model_fp32, inplace=False)
    
    print("model_fp32.qconfig: ", model_fp32.qconfig)
    print("qmodel.qconfig: ", qmodel.qconfig)
    print('Model prepared for QAT - check fake quantized nodes')
    print(qmodel)
    """
        LeNet5(
            (featureExtractor): Sequential(
                (0): ConvReLU2d(
                3, 6, kernel_size=(5, 5), stride=(1, 1)
                (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
                    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
                    (activation_post_process): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
                )
                (activation_post_process): FusedMovingAvgObsFakeQuantize(
                    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
                    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
                )
                )
                (1): Identity()
                (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
                (3): ConvReLU2d(
                6, 16, kernel_size=(5, 5), stride=(1, 1)
                (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
                    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
                    (activation_post_process): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
                )
                (activation_post_process): FusedMovingAvgObsFakeQuantize(
                    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
                    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
                )
                )
                (4): Identity()
                (5): AvgPool2d(kernel_size=2, stride=2, padding=0)
            )
            (classifier): Sequential(
                (0): Linear(
                in_features=400, out_features=120, bias=True
                (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
                    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
                    (activation_post_process): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
                )
                (activation_post_process): FusedMovingAvgObsFakeQuantize(
                    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
                    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
                )
                )
                (1): ReLU()
                (2): Linear(
                in_features=120, out_features=84, bias=True
                (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
                    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
                    (activation_post_process): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
                )
                (activation_post_process): FusedMovingAvgObsFakeQuantize(
                    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
                    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
                )
                )
                (3): ReLU()
                (4): Linear(
                in_features=84, out_features=10, bias=True
                (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
                    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
                    (activation_post_process): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
                )
                (activation_post_process): FusedMovingAvgObsFakeQuantize(
                    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
                    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
                )
                )
            )
            (quant): QuantStub(
                (activation_post_process): FusedMovingAvgObsFakeQuantize(
                fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
                (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
                )
            )
            (dequant): DeQuantStub()
            )
    """
    return qmodel

def train_one_epoch(model, best_loss, trainDataLoader, testDataLoader, device, writer, epochNum):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runningLoss = 0.0
    for i, data in enumerate(trainDataLoader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        runningLoss += loss

        # for every mini-batch
        if  i % 1000 == 999:
            runningVLoss = 0.0
            model.train(False) # disable gradient calc on validation set - IMPORTANT
            for j, vdata in enumerate(testDataLoader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs.to(device)) 
                vloss = criterion(voutputs, vlabels.to(device))
                runningVLoss += vloss
            
            (loss_t, acc_t)  = accuracy(model, trainDataLoader)
            (loss_v, acc_v)  = accuracy(model, testDataLoader)
            model.train(True)
            avg_loss = runningLoss / 1000.0
            avg_vloss = runningVLoss / len(testDataLoader)
            writer.add_scalars("Training vs Validation Loss", 
                                {"Training" : avg_loss, "Validation" : avg_vloss},
                                epochNum * len(trainDataLoader) + i
            )
            writer.add_scalars("Training vs Validation Accuracy", 
                                {"Training" : acc_t, "Validation" : acc_v},
                                epochNum * len(trainDataLoader) + i
            )
            writer.flush()
            print(f'[{epochNum}, {i + 1:5d}] loss: {runningLoss/1000:.3f} accuracy:T:{acc_t}, V:{acc_v}')
            
            if avg_vloss < best_loss:
                best_loss = avg_vloss
                model_path = 'model_{}_epoch={}.pth'.format(timestamp, epochNum)
                torch.save(model.state_dict(), model_path)
                # This saves weights. So when you want to load them, you need the python class. check lab6
            runningLoss = 0.0
    return best_loss


if __name__ == "__main__":

    (model_fp32_original, model_fp32) = step1(qat=True)
    
    (trainDataSet, trainDataLoader, testDataSet, testDataLoader, classes) = load_data("./../intro/data/cifar10/", writer=None)

    # Baseline accuracy: Comparing model acuracy before and after fuse model
    (vloss_fp32o, acc_fp32o) = accuracy(model_fp32_original, testDataLoader, withLoss=True)
    (vloss_fp32, acc_fp32) = accuracy(model_fp32, testDataLoader, withLoss=True)
    print ("1 LOSSES:   Model-1: ", vloss_fp32o, "Model-2: ", vloss_fp32)
    print ("1 ACCURACY: Model-1: ", acc_fp32o, "Model-2: ", acc_fp32)
    """
    # Baseline accuracy: Comparing model acuracy before and after fuse model
        SAME AS PTSQ
        MISSES:   Model-1:  4057 Model-2:  4057 TEST SET:  10000
        ACCURACY: Model-1:  0.5943 Model-2:  0.5943
    """

    # prepare for qat
    qmodel = step2(model_fp32)

    # Step 3 - compare models
    (vloss_fp32, acc_fp32) = accuracy(model_fp32, testDataLoader, withLoss=True)
    (vloss_q, acc_q) = accuracy(qmodel, testDataLoader, withLoss=True)
       
    print ("2 LOSSES:   Model-1: ", vloss_fp32, "Model-2: ", vloss_q)
    print ("2 ACCURACY: Model-1: ", acc_fp32, "Model-2: ", acc_q)
    """
        Q MODEL Accuracy drops
        LOSSES:   Model-1:  tensor(3284.2756, grad_fn=<AddBackward0>) Model-1:  tensor(3311.2471, grad_fn=<AddBackward0>)
        MISSES:   Model-1:  4042 Model-2:  4107 TEST SET:  10000
        ACCURACY: Model-1:  0.5958 Model-2:  0.5892999999999999
    """

    # Step 4 - QAT
    # tensorboard summary writer
    writer = SummaryWriter("./runs/experiment_1/")
    
    best_loss = 1_000_000
    best_loss = train_one_epoch(qmodel, best_loss, trainDataLoader, testDataLoader, device, writer, epochNum=1)
    """
    # QAT takes time and one needs to train over a few epochs.
    # Train and check accuracy after each epoch
    for nepoch in range(8):
        train_one_epoch(qmodel, trainDataLoader, testDataLoader, device, writer, epochNum=1)
        if nepoch > 3:
            # Freeze quantizer parameters
            qmodel.apply(torch.ao.quantization.disable_observer)
        if nepoch > 2:
            # Freeze batch norm mean and variance estimates
            qmodel.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # Check the accuracy after each epoch
    quantized_model = torch.ao.quantization.convert(qmodel.eval(), inplace=False)
    quantized_model.eval()
    
    print('Epoch %d :Evaluation accuracy on %d images, %2.2f'%(nepoch, num_eval_batches * eval_batch_size, top1.avg))
    """