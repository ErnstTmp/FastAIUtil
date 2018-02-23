
class BestLossSave(Callback):
    """ 
    Saves the best weigths with the best training loss
    Keeps the best loss accross several calls of fit.
    
    Pass to fit() with:
    
    cb = BestLossSave(learner, 'filename')
    ...
    fit( ..., callbacks=[cb])
    """
    
    def __init__(self, learner, filename=None, print_at_end=True):
        self.learner = learner
        self.epoch = 0
        self.best_loss = 1000000
        self.loss_sum = 0
        self.filename = filename if filename is not None else "best_loss"
        self.save_epoch = -1
        self.print_at_end = print_at_end
        self.num_batch = 0


    def on_batch_end(self, loss):
        self.loss_sum += loss
        self.num_batch += 1
    
    def on_epoch_end(self, vals):
        if self.best_loss > self.loss_sum:
            self.best_loss = self.loss_sum
            self.learner.save(self.filename)
            self.save_epoch = self.epoch
            
        self.epoch += 1
        self.loss_sum = 0
        self.max_num_batch = self.num_batch
        self.num_batch = 0
        
    def get_stats(self):
        return (self.epoch, self.save_epoch, self.best_loss/self.max_num_batch, self.filename)

    def print_stats(self):
        l = self.best_loss/self.max_num_batch
        print(f"BestLossSave: Total epochs: {self.epoch}, last saved in epoch: {self.save_epoch}, loss: {l}, filename: {self.filename}")

    def on_train_end(self):
        if self.print_at_end:
            self.print_stats()
