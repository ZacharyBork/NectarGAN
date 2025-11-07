# NectarGAN Examples - UNet (basic+vgg)
**Config: [config.json](config.json)**
## Test parameters:
| `block_type` | `lambda_gan` | `lambda_l1` | `lambda_l2` | `lambda_sobel` | `lambda_laplacian` | `lambda_vgg` |
| --- | --- | --- | --- | --- | --- | --- |
| ResidualUnetBlock | 1.0 | 100.0 | 0.0 | 0.0 | 0.0 | 10.0 |
## Loss Graphs
![Loss Graph](loss_graphs.png)
## Examples
| Epoch | {Input, Generated, Target} |
| --- | --- |
| 1 | ![Epoch 1 Example](examples/epoch1.png) |
| 20 | ![Epoch 1 Example](examples/epoch20.png) |
| 40 | ![Epoch 1 Example](examples/epoch40.png) |
| 60 | ![Epoch 1 Example](examples/epoch60.png) |
| 80 | ![Epoch 1 Example](examples/epoch80.png) |
| 100 | ![Epoch 1 Example](examples/epoch100.png) |
| 120 | ![Epoch 1 Example](examples/epoch120.png) |
| 140 | ![Epoch 1 Example](examples/epoch140.png) |
| 160 | ![Epoch 1 Example](examples/epoch160.png) |
| 180 | ![Epoch 1 Example](examples/epoch180.png) |
| 200 | ![Epoch 1 Example](examples/epoch200.png) |
