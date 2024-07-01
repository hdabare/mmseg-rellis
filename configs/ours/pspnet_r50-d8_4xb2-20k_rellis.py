_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/rellis.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
] # base config file which we build new config file on.
num_classses = 20
crop_size = (375, 600)
decode_head = dict(num_classes=num_classses)
auxiliary_head = dict(num_classes=num_classses)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor, 
             decode_head=decode_head, auxiliary_head=auxiliary_head)