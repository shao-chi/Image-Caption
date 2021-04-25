from torch.utils.tensorboard import SummaryWriter

class TensorBoard_Writer:
    def __init__(self, log_path):
        self.writer = SummaryWriter(log_path)

    def write_text(self, output, truths, step):
        self.writer.add_text('CAPTION/Sample_caption', output, step)
        self.writer.add_text('CAPTION/Ground_truth', truths, step)

    def write_batch(self, logs, step):
        for key, value in logs.items():
           self.writer.add_scalars(f'{key.upper()}/BATCH',
                                    {'Train': value['train'],
                                     'Valid': value['valid']}, step)

    def write_epoch(self, logs, epoch):
        for key, value in logs.items():
            if isinstance(value, dict):
                self.writer.add_scalars(f'{key.upper()}/EPOCH',
                                        {'Train': value['train'],
                                         'Valid': value['valid']}, epoch)
            else:
                self.writer.add_scalar(f'EVALUATION/{key.upper()}', value, epoch)

    def close(self):
        self.writer.close()