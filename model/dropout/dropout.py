from torch.nn.functional import dropout
from model.dropout.base import BaseDropout


class Dropout(BaseDropout):

    def __init__(self, dropout_prob=0.5):

        super(Dropout, self).__init__(dropout_prob)

    def apply_feature_mat(self, x, training=True):
        
        return dropout(x, self.dropout_prob, training=training)
    
    def apply_adj_mat(self, edge_index, edge_attr=None, training=True):

        return super(Dropout, self).apply_adj_mat(edge_index, edge_attr, training)
    
    def apply_message_mat(self, messages, training=True):

        return super(Dropout, self).apply_message_mat(messages, training)