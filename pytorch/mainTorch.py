from learner import Learner

if __name__ == '__main__':
    data_dir = '~/Dataset'
    learner = Learner(data_dir)
    learner.train()
    learner.test()
