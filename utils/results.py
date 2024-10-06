class Results:

    def __init__(self):

        '''
        Initialize an object to save the results in.
        '''
        
        self.results = None

    def update_results(self):

        '''
        Update the results object with train/val/test logs
            eg. predictions, labels, ...
        '''

        raise NotImplementedError

    def compute_metrics(self):

        '''
        Compute the metrics for train/val/test logs
            eg. metrics like errors, accuracy, f1-score, ...
        '''

        raise NotImplementedError

    def get(self, split):

        '''
        Get the metrics for one of train/val/test dataset.
        '''

        return self.results.get(split)