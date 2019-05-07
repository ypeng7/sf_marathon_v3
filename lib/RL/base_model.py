import tensorflow as tf
import numpy as np
import os
import shutil

# core config parameters
_gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
_initializer = tf.random_uniform_initializer(-0.1,  0.1)
_dropout_keep_prob = 0.6
_max_grad_norm = 0.5
_tensorboard_log_rel_path = "tb_log"
_save_kept_num = 10



class TfBaseConfig(object):
    @property
    def model_save_path(self):
        if self._model_save_path[-1] == "/":
            return self._model_save_path[:-1]
        return self._model_save_path

    def __init__(self, model_save_path, gpu_options=_gpu_options, initializer=_initializer,
                 keep_prob=_dropout_keep_prob,
                 max_grad_norm=_max_grad_norm, save_kept_num=_save_kept_num):
        self._model_save_path = model_save_path
        self.gpu_options = gpu_options
        self.initializer = initializer
        self.keep_prob = keep_prob
        self.max_grad_norm = max_grad_norm
        self.save_kept_num = save_kept_num
        self.tensorboard_log_path = self.model_save_path + "/" + _tensorboard_log_rel_path

class TfBaseModel(object):
    def __init__(self, config, global_reuse, *args, graph=None, sess=None, **kwargs):
        """
        can provide another model's Graph and Session. i.e. when reusing to create a test model, provide the train model's graph and session
        """
        self.error_list = []
        self.graph = tf.Graph() if graph is None else graph
        self.session = tf.Session(graph=self.graph) if graph is None else sess
        self.config = config
        self.tensorboard_watches = {}
        self.merged_summary = None
        self.global_reuse = global_reuse
        with self.graph.as_default(), tf.variable_scope("model", reuse=global_reuse):
            self.build(*args, **kwargs)
            self.saver = tf.train.Saver(max_to_keep=10)

        if not os.path.exists(self.config.tensorboard_log_path):
            os.makedirs(self.config.tensorboard_log_path)

        self.summary_writer = tf.summary.FileWriter(self.config.tensorboard_log_path)


    def build(self, *args, **kwargs):
        """
        TODO: build model
        :return:
        """
        return

    def back_propagate(self, cost, check_finite=False, optimizer=None, clip_grads=True, lr=None, momentum=None):
        """
        optimizer优先度：自定义optimizer > 如果lr=None就Adam > 如果lr<0就动态lr，注意需要feed self.lr > 固定lr的梯度下降
        """

        if optimizer is not None:
            bp_optimizer = optimizer
        elif lr is None:
            bp_optimizer = tf.train.AdamOptimizer()
        elif momentum is None:
            if lr < 0:
                lr = tf.placeholder(tf.float32, shape=(), name="adjustable_lr")
                self.lr = lr

            bp_optimizer = tf.train.GradientDescentOptimizer(lr)
        else:
            bp_optimizer = tf.train.MomentumOptimizer(lr, momentum)

        tvars = tf.trainable_variables()
        unclipped_grads =tf.gradients(cost, tvars)

        if check_finite:
            checked_grads = []
            for one_grad in unclipped_grads:
                grad_check = tf.logical_and(tf.is_finite(one_grad), tf.logical_not(tf.is_nan(one_grad)))
                checked_zero_grads = tf.where(grad_check, one_grad, tf.zeros(one_grad.shape))
                finite_unclipped_grads = checked_zero_grads
                checked_grads.append(finite_unclipped_grads)
        else:
            checked_grads = unclipped_grads

        if clip_grads:
            grads, _ = tf.clip_by_global_norm(checked_grads, self.config.max_grad_norm)
        else:
            grads = unclipped_grads

        self.eval_op = bp_optimizer.apply_gradients(zip(grads, tvars))


    def initialize_variables(self, assign_dict=None):
        """
        :param transfer_dict: {variable_reference/variable_name_with_scope_name : tensor_value, ......}
                                key can be string, or a tf tensor reference
        :return:
        """
        with self.graph.as_default():
            tf.global_variables_initializer().run(session=self.session)

            if assign_dict is not None:
                ref_assign_dict = {}

                for k, v in assign_dict.items():
                    if isinstance(k, str):
                        tf_vars = tf.trainable_variables()
                        matching_refs = list(filter(lambda x: x.name.__contains__(k), tf_vars))
                        if len(matching_refs) == 1:
                            ref_assign_dict[matching_refs[0]] = v
                        else:
                            print("transfer variable not found in tf variables, or found with more than one variable containing the key, please check the variable names")
                    else:
                        ref_assign_dict[k] = v


                assign_ops = [tf.assign(x, y) for x, y in ref_assign_dict.items()]
                self.session.run(assign_ops)
        return

    def add_to_tensorboard(self, var, watch_name):
        with tf.name_scope("summaries/" + watch_name):
            if isinstance(var, tf.Tensor):
                var_tensor = var
                delay_assign = False
            else:
                if isinstance(var, int):
                    var = float(var)
                var_tensor = tf.get_variable("summary_"+watch_name, initializer=var, dtype=tf.float32)
                delay_assign = True

            type_str = "scalar" if var_tensor.shape.ndims == 0 else "histogram"

            if type_str == "histogram":
                summary_var =  tf.summary.histogram(watch_name, var_tensor)
            else:
                summary_var = tf.summary.scalar(watch_name, var_tensor)

            self.tensorboard_watches[watch_name] = (var_tensor, summary_var, delay_assign)


    def save_best_models(self, iter_num, error, folder_path=None):
        if folder_path is None:
            folder_path = self.config.model_save_path

        def save_best():
            with self.graph.as_default():
                self.saver.save(self.session, folder_path + "/model_"+str(iter_num) + "_"+str(error) + "_.ckpt")

        if not os.path.exists(folder_path + "/measure_record.txt"):
            self.error_lsit = []
            measure_recorder = open(folder_path + "/measure_record.txt", mode="w")
        else:
            fr = open(folder_path + "/measure_record.txt", mode="r")
            lines = fr.read().split("\n")
            fr.close()
            self.error_list = [(int(line.split("\t")[0]), float(line.split("\t")[1])) for line in lines if line != ""]
            measure_recorder = open(folder_path + "/measure_record.txt", mode="w")

        if len(self.error_list) < self.config.save_kept_num:
            save_best()
            self.error_list.append((iter_num, error))
        elif error < max([x[1] for x in self.error_list]):
            threashold = max([x[1] for x in self.error_list])
            self.error_list = list(filter(lambda x: x[1] < threashold, self.error_list))
            save_best()
            self.error_list.append((iter_num, error))
        else:
            pass

        self.error_list = list(sorted(self.error_list, key=lambda x: x[1]))
        measure_recorder.write("\n".join([str(x)+"\t"+str(y) for x, y in self.error_list]))
        measure_recorder.close()
        return



    def load_model(self, iter_num, folder_path=None):

        if folder_path is None:
            folder_path = self.config.model_save_path

        import os
        file_list = os.listdir(folder_path)
        candidate_files = list(filter(lambda x: (len(x.split("_")) > 2) and int((x.split("_"))[1])==iter_num, file_list))
        evaluate_measure = candidate_files[0].split("_")[2]
        with self.graph.as_default():
            self.saver.restore(self.session, folder_path+"/model_"+str(iter_num)+"_"+str(evaluate_measure)+"_.ckpt")

        return

    def process_batch(self, iter_num, feed_dict, *args, segmented_num=None, tensors_requires_segmentation=None, tensorboard_extra_assigns=None):
        """
        :param feed_dict: feed_dict for session.run
        :param args: running tf variables
        :param segmented_num: whether feed batch need to be segmented into smaller batches. int
        :param tensors_requires_segmentation: for each value in feed_dict, indicate whether it should be segmented. list of bool
        :param tensorboard_extra_assigns: dict with key=watch_name, val=assign_value for extra tensorboard watches in runtime
        :return:
        """

        if segmented_num is not None:
            assert len(tensors_requires_segmentation) == len(feed_dict)

        if self.merged_summary is None and len(self.tensorboard_watches) > 0:
            merge_item_list = [x[1] for x in self.tensorboard_watches.values()]
            self.merged_summary = tf.summary.merge(merge_item_list)
        if tensorboard_extra_assigns is not None:
            assign_item_list = [(self.tensorboard_watches[kv[0]][0], kv[1]) for kv in tensorboard_extra_assigns.items()
                                if kv[0] in self.tensorboard_watches and self.tensorboard_watches[kv[0]][2]]
            self.session.run([tf.assign(x[0], x[1]) for x in assign_item_list])


        def run_with_merge(_args, feeds):
            if self.merged_summary is None:
                return self.session.run(args, feed_dict=feeds)
            else:
                run_li = list(args)
                run_li.append(self.merged_summary)
                results = self.session.run(run_li, feed_dict=feeds)
                runned_merged = results[-1]
                self.summary_writer.add_summary(runned_merged, iter_num)
                if iter_num > 0 and iter_num % 20 == 0:
                    self.summary_writer.reopen()
                tensor_results = results[:-1]
                return tensor_results

        with self.graph.as_default():
            if segmented_num is None:
                return run_with_merge(args, feed_dict)
            else:
                total_sample_num = (list(filter(lambda x: x[0], zip(tensors_requires_segmentation, list(feed_dict.values()))))[0])[1].shape[0]
                mini_batch_num = total_sample_num // segmented_num
                residual_batch_num = total_sample_num % segmented_num

                results_list = [[] for _ in range(len(args))]
                for batch_i in range(mini_batch_num):
                    feed_dict_mini = dict([(x, y[segmented_num*batch_i:segmented_num*(batch_i+1)]) for x, y in feed_dict.items()])
                    results_mini = run_with_merge(args, feed_dict_mini)
                    for ri in range(len(results_list)):
                        results_list[ri].append(results_mini[ri])

                if residual_batch_num > 0:
                    feed_dict_mini = dict([(x, y[segmented_num*mini_batch_num:]) for x, y in feed_dict.items()])
                    results_mini = self.session.run(args, feed_dict=feed_dict_mini)
                    for ri in range(len(results_list)):
                        results_list[ri].append(results_mini[ri])

                concatnated_results = [np.concatenate(x, axis=0) for x in results_list]
                return tuple(concatnated_results)


import random
class RandomSampler(object):

    @property
    def item_list(self):
        return self._item_list

    def __init__(self, item_list, shuffle=True):
        self._item_list = item_list
        self._pointer = 0
        self._shuffle = shuffle
        if shuffle:
            random.shuffle(self._item_list)


    def get_batch(self, batch_size, func_on_each, *args, **kwargs):
        batch_result = []
        while(len(batch_result) < batch_size):

            result = func_on_each(self._item_list[self._pointer], *args, **kwargs)
            if result is not None:
                batch_result.append(result)

            self._pointer +=1
            if self._pointer >= len(self._item_list):
                self._pointer = 0
                if self._shuffle:
                    random.shuffle(self._item_list)

        return batch_result
