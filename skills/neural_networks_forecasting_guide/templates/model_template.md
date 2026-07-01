# Model Template

Every model should inherit

torch.nn.Module

Implement

__init__()

forward()

Provide

save()

load()

Support

device transfer

evaluation mode

training mode

Document

expected tensor shapes

Example

Input

(batch,
 input_length,
 features)

Output

(batch,
 horizon,
 targets)

Do not hardcode tensor dimensions.