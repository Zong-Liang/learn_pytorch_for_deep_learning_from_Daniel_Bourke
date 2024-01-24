# 02_pytrorch_workflow

![01_a_pytorch_workflow](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main//202401202225286.png)

- prepare data
- build or pick a pretrained model to suit your problem
  - pick a loss function and optimizer
  - build a training loop
    - `model.train()`
    - forward pass
    - calculate loss
    - optimizer zero grad
    - loss backward
    - optimizer step
    - `model.eval()`
    - `with torch.inference_mode()`
    - forward pass
    - calculate loss
- fit the model to the data and make a prediction
- evaluate the model
- improve through experimentation
- save and reload your trained model