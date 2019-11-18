import logging
import pprint
import sys
import datetime
import torch
import utils
import matplotlib.pyplot as plt
import time
from sklearn.metrics import average_precision_score
from scipy.sparse import coo_matrix
import numpy as np




class Logger():
    def __init__(self, args, num_classes, minibatch_log_interval=10):

        if args is not None:
            currdate=str(datetime.datetime.today().strftime('%Y%m%d%H%M%S'))
            self.log_name= 'log/log_'+args.data+'_'+args.task+'_'+args.model+'_'+currdate+'_r'+str(args.rank)+'.log'

            if args.use_logfile:
                print ("Log file:", self.log_name)
                logging.basicConfig(filename=self.log_name, level=logging.INFO)
            else:
                print ("Log: STDOUT")
                logging.basicConfig(stream=sys.stdout, level=logging.INFO)

            logging.info ('*** PARAMETERS ***')
            logging.info (pprint.pformat(args.__dict__)) # displays the string
            logging.info ('')
        else:
            print ("Log: STDOUT")
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        self.num_classes = num_classes
        self.minibatch_log_interval = minibatch_log_interval
        self.eval_k_list = [10, 100, 1000]
        self.args = args


    def get_log_file_name(self):
        return self.log_name

    def log_epoch_start(self, epoch, num_minibatches, set, minibatch_log_interval=None):
        #ALDO
        self.epoch = epoch
        ######
        self.set = set
        self.losses = []
        self.errors = []
        self.MRRs = []
        self.MAPs = []
        #self.time_step_sizes = []
        self.conf_mat_tp = {}
        self.conf_mat_fn = {}
        self.conf_mat_fp = {}
        self.conf_mat_tp_at_k = {}
        self.conf_mat_fn_at_k = {}
        self.conf_mat_fp_at_k = {}
        for k in self.eval_k_list:
            self.conf_mat_tp_at_k[k] = {}
            self.conf_mat_fn_at_k[k] = {}
            self.conf_mat_fp_at_k[k] = {}

        for cl in range(self.num_classes):
            self.conf_mat_tp[cl]=0
            self.conf_mat_fn[cl]=0
            self.conf_mat_fp[cl]=0
            for k in self.eval_k_list:
                self.conf_mat_tp_at_k[k][cl]=0
                self.conf_mat_fn_at_k[k][cl]=0
                self.conf_mat_fp_at_k[k][cl]=0

        if self.set == "TEST":
            self.conf_mat_tp_list = {}
            self.conf_mat_fn_list = {}
            self.conf_mat_fp_list = {}
            for cl in range(self.num_classes):
                self.conf_mat_tp_list[cl]=[]
                self.conf_mat_fn_list[cl]=[]
                self.conf_mat_fp_list[cl]=[]

        self.batch_sizes=[]
        self.minibatch_done = 0
        self.num_minibatches = num_minibatches
        if minibatch_log_interval is not None:
            self.minibatch_log_interval = minibatch_log_interval
        logging.info('################ '+set+' epoch '+str(epoch)+' ###################')
        self.lasttime = time.monotonic()
        self.ep_time = self.lasttime

    def log_minibatch(self, predictions, true_classes, loss, **kwargs):

        probs = torch.softmax(predictions,dim=1)[:,1]
        if self.set in ['TEST', 'VALID'] and self.args.task == 'link_pred':
            MRR = self.get_MRR(probs,true_classes, kwargs['adj'],do_softmax=False)
        else:
            MRR = torch.tensor([0.0])

        MAP = torch.tensor(self.get_MAP(probs,true_classes, do_softmax=False))

        error, conf_mat_per_class = self.eval_predicitions(predictions, true_classes, self.num_classes)
        conf_mat_per_class_at_k={}
        for k in self.eval_k_list:
            conf_mat_per_class_at_k[k] = self.eval_predicitions_at_k(predictions, true_classes, self.num_classes, k)

        batch_size = predictions.size(0)
        self.batch_sizes.append(batch_size)

        self.losses.append(loss) #loss.detach()
        self.errors.append(error)
        self.MRRs.append(MRR)
        self.MAPs.append(MAP)
        for cl in range(self.num_classes):
            self.conf_mat_tp[cl]+=conf_mat_per_class.true_positives[cl]
            self.conf_mat_fn[cl]+=conf_mat_per_class.false_negatives[cl]
            self.conf_mat_fp[cl]+=conf_mat_per_class.false_positives[cl]
            for k in self.eval_k_list:
                self.conf_mat_tp_at_k[k][cl]+=conf_mat_per_class_at_k[k].true_positives[cl]
                self.conf_mat_fn_at_k[k][cl]+=conf_mat_per_class_at_k[k].false_negatives[cl]
                self.conf_mat_fp_at_k[k][cl]+=conf_mat_per_class_at_k[k].false_positives[cl]
            if self.set == "TEST":
                self.conf_mat_tp_list[cl].append(conf_mat_per_class.true_positives[cl])
                self.conf_mat_fn_list[cl].append(conf_mat_per_class.false_negatives[cl])
                self.conf_mat_fp_list[cl].append(conf_mat_per_class.false_positives[cl])

        self.minibatch_done+=1
        if self.minibatch_done%self.minibatch_log_interval==0:
            mb_error = self.calc_epoch_metric(self.batch_sizes, self.errors)
            mb_MRR = self.calc_epoch_metric(self.batch_sizes, self.MRRs)
            mb_MAP = self.calc_epoch_metric(self.batch_sizes, self.MAPs)
            partial_losses = torch.stack(self.losses)
            logging.info(self.set+ ' batch %d / %d - partial error %0.4f - partial loss %0.4f - partial MRR  %0.4f - partial MAP %0.4f' % (self.minibatch_done, self.num_minibatches, mb_error, partial_losses.mean(), mb_MRR, mb_MAP))

            tp=conf_mat_per_class.true_positives
            fn=conf_mat_per_class.false_negatives
            fp=conf_mat_per_class.false_positives
            logging.info(self.set+' batch %d / %d -  partial tp %s,fn %s,fp %s' % (self.minibatch_done, self.num_minibatches, tp, fn, fp))
            precision, recall, f1 = self.calc_microavg_eval_measures(tp, fn, fp)
            logging.info (self.set+' batch %d / %d - measures partial microavg - precision %0.4f - recall %0.4f - f1 %0.4f ' % (self.minibatch_done, self.num_minibatches, precision,recall,f1))
            for cl in range(self.num_classes):
                cl_precision, cl_recall, cl_f1 = self.calc_eval_measures_per_class(tp, fn, fp, cl)
                logging.info (self.set+' batch %d / %d - measures partial for class %d - precision %0.4f - recall %0.4f - f1 %0.4f ' % (self.minibatch_done, self.num_minibatches, cl,cl_precision,cl_recall,cl_f1))

            logging.info (self.set+' batch %d / %d - Batch time %d ' % (self.minibatch_done, self.num_minibatches, (time.monotonic()-self.lasttime) ))

        self.lasttime=time.monotonic()

    def log_epoch_done(self):
        eval_measure = 0

        self.losses = torch.stack(self.losses)
        logging.info(self.set+' mean losses '+ str(self.losses.mean()))
        if self.args.target_measure=='loss' or self.args.target_measure=='Loss':
            eval_measure = self.losses.mean()

        epoch_error = self.calc_epoch_metric(self.batch_sizes, self.errors)
        logging.info(self.set+' mean errors '+ str(epoch_error))

        epoch_MRR = self.calc_epoch_metric(self.batch_sizes, self.MRRs)
        epoch_MAP = self.calc_epoch_metric(self.batch_sizes, self.MAPs)
        logging.info(self.set+' mean MRR '+ str(epoch_MRR)+' - mean MAP '+ str(epoch_MAP))
        if self.args.target_measure=='MRR' or self.args.target_measure=='mrr':
            eval_measure = epoch_MRR
        if self.args.target_measure=='MAP' or self.args.target_measure=='map':
            eval_measure = epoch_MAP

        logging.info(self.set+' tp %s,fn %s,fp %s' % (self.conf_mat_tp, self.conf_mat_fn, self.conf_mat_fp))
        precision, recall, f1 = self.calc_microavg_eval_measures(self.conf_mat_tp, self.conf_mat_fn, self.conf_mat_fp)
        logging.info (self.set+' measures microavg - precision %0.4f - recall %0.4f - f1 %0.4f ' % (precision,recall,f1))
        if str(self.args.target_class) == 'AVG':
            if self.args.target_measure=='Precision' or self.args.target_measure=='prec':
                eval_measure = precision
            elif self.args.target_measure=='Recall' or self.args.target_measure=='rec':
                eval_measure = recall
            else:
                eval_measure = f1


        for cl in range(self.num_classes):
            cl_precision, cl_recall, cl_f1 = self.calc_eval_measures_per_class(self.conf_mat_tp, self.conf_mat_fn, self.conf_mat_fp, cl)
            logging.info (self.set+' measures for class %d - precision %0.4f - recall %0.4f - f1 %0.4f ' % (cl,cl_precision,cl_recall,cl_f1))
            if str(cl) == str(self.args.target_class):
                if self.args.target_measure=='Precision' or self.args.target_measure=='prec':
                    eval_measure = cl_precision
                elif self.args.target_measure=='Recall' or self.args.target_measure=='rec':
                    eval_measure = cl_recall
                else:
                    eval_measure = cl_f1

        for k in self.eval_k_list: #logging.info(self.set+' @%d tp %s,fn %s,fp %s' % (k, self.conf_mat_tp_at_k[k], self.conf_mat_fn_at_k[k], self.conf_mat_fp_at_k[k]))
            precision, recall, f1 = self.calc_microavg_eval_measures(self.conf_mat_tp_at_k[k], self.conf_mat_fn_at_k[k], self.conf_mat_fp_at_k[k])
            logging.info (self.set+' measures@%d microavg - precision %0.4f - recall %0.4f - f1 %0.4f ' % (k,precision,recall,f1))

            for cl in range(self.num_classes):
                cl_precision, cl_recall, cl_f1 = self.calc_eval_measures_per_class(self.conf_mat_tp_at_k[k], self.conf_mat_fn_at_k[k], self.conf_mat_fp_at_k[k], cl)
                logging.info (self.set+' measures@%d for class %d - precision %0.4f - recall %0.4f - f1 %0.4f ' % (k, cl,cl_precision,cl_recall,cl_f1))


        logging.info (self.set+' Total epoch time: '+ str(((time.monotonic()-self.ep_time))))

        return eval_measure

    def get_MRR(self,predictions,true_classes, adj ,do_softmax=False):
        if do_softmax:
            probs = torch.softmax(predictions,dim=1)[:,1]
        else:
            probs = predictions

        probs = probs.cpu().numpy()
        true_classes = true_classes.cpu().numpy()
        adj = adj.cpu().numpy()

        pred_matrix = coo_matrix((probs,(adj[0],adj[1]))).toarray()
        true_matrix = coo_matrix((true_classes,(adj[0],adj[1]))).toarray()

        row_MRRs = []
        for i,pred_row in enumerate(pred_matrix):
            #check if there are any existing edges
            if np.isin(1,true_matrix[i]):
                row_MRRs.append(self.get_row_MRR(pred_row,true_matrix[i]))

        avg_MRR = torch.tensor(row_MRRs).mean()
        return avg_MRR

    def get_row_MRR(self,probs,true_classes):
        existing_mask = true_classes == 1
        #descending in probability
        ordered_indices = np.flip(probs.argsort())

        ordered_existing_mask = existing_mask[ordered_indices]

        existing_ranks = np.arange(1,
                                   true_classes.shape[0]+1,
                                   dtype=np.float)[ordered_existing_mask]

        MRR = (1/existing_ranks).sum()/existing_ranks.shape[0]
        return MRR


    def get_MAP(self,predictions,true_classes, do_softmax=False):
        if do_softmax:
            probs = torch.softmax(predictions,dim=1)[:,1]
        else:
            probs = predictions

        predictions_np = probs.detach().cpu().numpy()
        true_classes_np = true_classes.detach().cpu().numpy()

        return average_precision_score(true_classes_np, predictions_np)

    def eval_predicitions(self, predictions, true_classes, num_classes):
        predicted_classes = predictions.argmax(dim=1)
        failures = (predicted_classes!=true_classes).sum(dtype=torch.float)
        error = failures/predictions.size(0)

        conf_mat_per_class = utils.Namespace({})
        conf_mat_per_class.true_positives = {}
        conf_mat_per_class.false_negatives = {}
        conf_mat_per_class.false_positives = {}

        for cl in range(num_classes):
            cl_indices = true_classes == cl

            pos = predicted_classes == cl
            hits = (predicted_classes[cl_indices] == true_classes[cl_indices])

            tp = hits.sum()
            fn = hits.size(0) - tp
            fp = pos.sum() - tp

            conf_mat_per_class.true_positives[cl] = tp
            conf_mat_per_class.false_negatives[cl] = fn
            conf_mat_per_class.false_positives[cl] = fp
        return error, conf_mat_per_class


    def eval_predicitions_at_k(self, predictions, true_classes, num_classes, k):
        conf_mat_per_class = utils.Namespace({})
        conf_mat_per_class.true_positives = {}
        conf_mat_per_class.false_negatives = {}
        conf_mat_per_class.false_positives = {}

        if predictions.size(0)<k:
            k=predictions.size(0)

        for cl in range(num_classes):
            # sort for prediction with higher score for target class (cl)
            _, idx_preds_at_k = torch.topk(predictions[:,cl], k, dim=0, largest=True, sorted=True)
            predictions_at_k = predictions[idx_preds_at_k]
            predicted_classes = predictions_at_k.argmax(dim=1)

            cl_indices_at_k = true_classes[idx_preds_at_k] == cl
            cl_indices = true_classes == cl

            pos = predicted_classes == cl
            hits = (predicted_classes[cl_indices_at_k] == true_classes[idx_preds_at_k][cl_indices_at_k])

            tp = hits.sum()
            fn = true_classes[cl_indices].size(0) - tp # This only if we want to consider the size at K -> hits.size(0) - tp
            fp = pos.sum() - tp

            conf_mat_per_class.true_positives[cl] = tp
            conf_mat_per_class.false_negatives[cl] = fn
            conf_mat_per_class.false_positives[cl] = fp
        return conf_mat_per_class


    def calc_microavg_eval_measures(self, tp, fn, fp):
        tp_sum = sum(tp.values()).item()
        fn_sum = sum(fn.values()).item()
        fp_sum = sum(fp.values()).item()

        p = tp_sum*1.0 / (tp_sum+fp_sum)
        r = tp_sum*1.0 / (tp_sum+fn_sum)
        if (p+r)>0:
            f1 = 2.0 * (p*r) / (p+r)
        else:
            f1 = 0
        return p, r, f1

    def calc_eval_measures_per_class(self, tp, fn, fp, class_id):
        #ALDO
        if type(tp) is dict:
            tp_sum = tp[class_id].item()
            fn_sum = fn[class_id].item()
            fp_sum = fp[class_id].item()
        else:
            tp_sum = tp.item()
            fn_sum = fn.item()
            fp_sum = fp.item()
        ########
        if tp_sum==0:
            return 0,0,0

        p = tp_sum*1.0 / (tp_sum+fp_sum)
        r = tp_sum*1.0 / (tp_sum+fn_sum)
        if (p+r)>0:
            f1 = 2.0 * (p*r) / (p+r)
        else:
            f1 = 0
        return p, r, f1

    def calc_epoch_metric(self,batch_sizes, metric_val):
        batch_sizes = torch.tensor(batch_sizes, dtype = torch.float)
        epoch_metric_val = torch.stack(metric_val).cpu() * batch_sizes
        epoch_metric_val = epoch_metric_val.sum()/batch_sizes.sum()

        return epoch_metric_val.detach().item()
