import pathlib

from pix2pix_graphical.ui.run import Interface

{
    # from .worker import TrainerWorker

    # def update_loss():
    #     print('update_loss')

    # def update_images():
    #     print('update_images')

    # def start_training():
    #     config = json.load(open('./config.json'))
    #     worker = TrainerWorker(config)
    #     worker.loss_updated.connect(update_loss)
    #     worker.image_updated.connect(update_images)
    #     worker.start()
}

if __name__ == "__main__":
    interface = Interface()
    interface.run()

