import datetime
import os, time
import numpy as np
import multiprocessing as mp
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import tensorflow as tf
import env
import a3c
import load_trace
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_till_video_end
S_LEN = 8  # take how many frames in the past 取过去的多少帧
A_DIM = 6
ACTOR_LR_RATE = 0.0001  #0.0001
CRITIC_LR_RATE = 0.001  #0.001
NUM_AGENTS = 8
TRAIN_SEQ_LEN = 100  # take as a train  100次训练为1批
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 297.0
M_IN_K = 1000.0
REBUF_PENALTY = 1.85  # 1 sec rebuffering -> 3 Mbps #tingdun chengfa
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent 默认视频质量，无代理
RANDOM_SEED = 42
RAND_RANGE = 1000   #1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = './cooked_traces/'
# NN_MODEL = './results/pretrain_linear_reward.ckpt'
NN_MODEL = None

def testing(epoch, nn_model, log_file):
    # clean up the test results folder 清理测试结果文件夹
    os.system('rm -r ' + TEST_LOG_FOLDER)
    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)
    # run test script 运行测试脚本
    os.system('python3 rl_test.py ' + nn_model)
    
    # append test performance to the log将测试性能附加到日志中
    rewards, entropies = [], []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward, entropy = [], []
        with open(TEST_LOG_FOLDER + test_log_file, 'r') as f:
            for line in f:
                parse = line.split()
                try:
                    entropy.append(float(parse[-2]))
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.sum(reward[1:]))
        entropies.append(np.mean(entropy[1:]))

    rewards = np.array(rewards)
    #print(rewards)
    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()

    return rewards_mean, np.mean(entropies)


def central_agent(net_params_queues, exp_queues):
    #params参数
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    #交代log的地址以及格式，用于打印运行信息
    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    #Session提供了 Operation 执行和 Tensor 值的环境，使用with是为了不手动关闭会话
    with tf.compat.v1.Session() as sess, open(LOG_FILE + '_test', 'w') as test_log_file:

        #创建actor\critic
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        #??使用tensorboard，a3c.build_summaries() 的定义就是在tensorboard上打印 td_loss eps_total_reward avg_entropy
        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.compat.v1.global_variables_initializer())  #相当于对所有的都初始化
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        TRAIN_SUMMARY_DIR = './results/'+curr_time+'/train'
        TEST_SUMMARY_DIR = './results/'+curr_time+'/test'
        writer = tf.compat.v1.summary.FileWriter(TRAIN_SUMMARY_DIR, sess.graph)  # training monitor训练监控
        test_writer = tf.compat.v1.summary.FileWriter(TEST_SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.compat.v1.train.Saver()  # save neural net parameters保存神经网络参数

        # restore neural net parameters恢复神经网络参数
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file nn模型是文件的路径
            saver.restore(sess, nn_model)
            print("Model restored.")

        epoch = 0
        test_avg_reward, test_avg_entropy, test_avg_td_loss = 0, 0.5, 0

        # assemble experiences from agents, compute the gradients收集agents的经验，计算梯度
        #这是一个同步的过程，每个epoch一次，将coordinator的数据传入到每个agent中
        while True:
            # synchronize the network parameters of work agent 同步作业agent的网络参数
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])
                # Note: this is synchronous version of the parallel training, # 这是并行训练的同步版本
                # which is easier to understand and probe(#调查). The framework can be
                # fairly easily modified to support asynchronous training.
                #lv: 可以相当容易地修改框架以支持异步训练。异步训练的一些实践(核心是无锁SGD)得到了很好的解释
                # Some practices of asynchronous training (lock-free SGD at
                # its core) are nicely explained in the following two papers:
                # https://arxiv.org/abs/1602.01783
                # https://arxiv.org/abs/1106.5730

            # record average reward and td loss change
            # in the experiences from the agents记录个体经验中平均收益和td损失的变化
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0 

            # assemble experiences from the agents 从agent那里收集经验
            actor_gradient_batch = []
            critic_gradient_batch = []

            for i in range(NUM_AGENTS):
                # 注意这里的get是一个阻塞函数，意思是如果没有get到就会一直挂起等待，所以实现了一个同步的过程
                # 换句话说，就是要等到每一个agent把参数传给coordinator，才会继续往下
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()
                #通过获取到的刚刚的参数进行计算，获取到对应的gradient值
                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])

            # compute aggregated gradient 计算聚合梯度
            assert NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)
            #将得到的gradient更新每个agent
            for i in range(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])
                critic.apply_gradients(critic_gradient_batch[i])

            # log training information记录培训信息/打印并更新tensorboard上的信息
            epoch += 1
            avg_reward = total_reward  / total_agents
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len

            logging.info('Epoch: ' + str(epoch) +
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward: ' + str(avg_reward) +
                         ' Avg_entropy: ' + str(avg_entropy))

            #Training summary训练总结
            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: avg_entropy
            })
            writer.add_summary(summary_str, epoch)
            writer.flush()
            # Testing summary 测试总结
            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: test_avg_td_loss,
                summary_vars[1]: test_avg_reward,
                summary_vars[2]: test_avg_entropy
            })

            test_writer.add_summary(summary_str, epoch)
            test_writer.flush()

            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.将神经网络参数写入磁盘
                # 保存模型，并调用testing
                # 没有break意味着需要手动跳出循环，否则无限次运行
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt")
                logging.info("Model saved in file: " + save_path)
                test_avg_reward, test_avg_entropy = testing(epoch, SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt", test_log_file)

def agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue):
    #创建agent的训练模型
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=agent_id)

    with tf.compat.v1.Session() as sess, open(LOG_FILE + '_agent_' + str(agent_id), 'w') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        # initial synchronization(同步) of the network parameters from the coordinator从协调器初始同步网络参数
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        time_stamp = 0
        while True:  # experience video streaming forever永远体验视频流

            # the action is from the last decision行动源于上一步的决定
            # this is to make the framework similar to the real 这是为了使框架与实际相似
            #模拟真实的信息，相当于 download video chunk over mahimahi
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)
            #print(video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain)
            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # -- linear reward --
            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            # -- log scale reward对数尺度奖励 --
            # log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[-1]))
            # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[-1]))

            # reward = log_bit_rate \
            #          - REBUF_PENALTY * rebuf \
            #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

            # -- HD reward --
            # reward = HD_REWARD[bit_rate] \
            #          - REBUF_PENALTY * rebuf \
            #          - SMOOTH_PENALTY * np.abs(HD_REWARD[bit_rate] - HD_REWARD[last_bit_rate])

            r_batch.append(reward)

            last_bit_rate = bit_rate

            # retrieve previous state检索上一状态
            if len(s_batch) == 0:
                state = np.zeros((S_INFO, S_LEN))
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record出队列历史记录
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            #print(state[5, -1])
            # compute action probability vector计算动作概率向量
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)      #这是计算cdf
            #这个是action的概率求cdf，然后随机生成一个值（0-1）,返回第一个最大的值（返回第一个True的下标），具体见下
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states
            #注意:我们需要将概率离散成1/RAND_RANGE步骤，因为在传递单个状态和批处理状态时存在内在的差异
            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            # report experience to the coordinator 达到100向协调器报告经验一次
            if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it 忽略第一个数据块，因为我们无法控制它
                               end_of_video,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator从协调器同步网络参数
                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]

                log_file.write('\n')  # so that in the log we know where video ends这样我们就能在日志中知道视频的结尾

            # store the state and action into batches将状态和操作批量存储
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here使用这里的默认操作

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)

            else:
                s_batch.append(state)

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1
                a_batch.append(action_vec)


def main():

    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM

    # create result directory创建结果目录
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # inter-process communication queues创建16个队列，用于进程间的通信
    #将队列存储到net_params_queues和exp_queues中
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes创建一个协调器和多个代理进程
    # (note: threading is not desirable due to python GIL)(注意:由于python GIL，线程是不可取的)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
    agents = []
    #创建并开启16个子process，这些线程用于异步计算（虽然这里的代码是同步）
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, all_cooked_time, all_cooked_bw,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done阻塞住主进程再等待子进程结束，然后再往下执行,（里面会用wait()）
    coordinator.join()


if __name__ == '__main__':
    main()
