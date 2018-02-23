class BestValSave(Callback):
    """ 
    Saves the best weigths with the best validation, either based on loss or metrics (if available) 
    Keeps the best loss/metrics across calls of fit.
    Makeing predictions using the net with the best validation loss/metrics is a 
    (simple but not very elegant) regularization technique
    
        savetype can be 'loss' or 'metrics'
        print_at_end determines whether information about the saved log is printed.
    Pass to fit() with:
    
    cb = BestValSave(learner, 'filename')
    ...
    fit( ..., callbacks=[cb])
    """
    
    def __init__(self, learner, filename=None, savetype='loss', print_at_end=True):
        self.learner = learner
        self.epoch = 0
        self.best_val = 1000000
        self.filename = filename if filename is not None else "best_val"
        self.valtype = 1 if savetype == 'metrics' else 0  
        self.save_epoch = -1
        self.print_at_end = print_at_end
        
    def on_epoch_end(self, vals):
        val = vals[self.valtype] # validation loss: vals[0], metrics: vals[1:]
        if self.best_val > val:
            self.best_val = val
            learner.save(self.filename)
            self.save_epoch = self.epoch
            
        self.epoch += 1

    def get_stats(self):
        return (self.epoch, self.save_epoch, self.filename)
    
    def get_stats(self):
        return (self.epoch, self.save_epoch, self.best_val, self.filename)

    def print_stats(self):
        print(f"BestValSave: Total epochs: {self.epoch}, last saved in epoch: {self.save_epoch}, metrics: {self.best_val}, filename: {self.filename}")
        
    def on_train_end(self):
        if self.print_at_end:
            self.print_stats()
