import matplotlib.pyplot as plt
import numpy as np

accuracy = [
    0.4482,
    0.4361,
    0.4530,
    0.4458,
    0.4795,
    0.5542,
    0.6193,
    0.6578,
    0.6289,
    0.5928,
    0.6048,
    0.6337,
    0.6410
]

loss = [
    1.4245,
    1.2751,
    1.1762,
    1.1121,
    1.0656,
    1.0020,
    1.0027,
    0.9466,
    0.9201,
    0.8682,
    0.8701,
    0.8188,
    0.7604
]

val_loss = [
    1.5884,
    1.5599,
    1.4856,
    1.5683,
    1.4505,
    1.4959,
    1.4418,
    1.4887,
    1.3404,
    1.3959,
    1.2806,
    1.2677,
    1.3854
]

val_acc = [
    0.4022,
    0.4078,
    0.3520,
    0.2626,
    0.2682,
    0.3464,
    0.3520,
    0.1397,
    0.3017,
    0.3184,
    0.0279,
    0.3017,
    0.2123
]


plt.figure(num=1)
plt.xlabel('epoch')
plt.plot(accuracy)
plt.plot(loss)
plt.plot(val_loss)
plt.plot(val_acc)
plt.legend(['accuracy','loss', 'val_loss', 'val_acc'],loc='center left')
plt.show()
