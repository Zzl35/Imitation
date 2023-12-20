from tqdm import tqdm

from trainer.base import RLTrainer


class OfflineRLTrainer(RLTrainer):
    def __init__(self, train_env, eval_env, expert_buffer, policy_buffer, logger):
        super().__init__(train_env, eval_env, expert_buffer, policy_buffer, logger)

    def train(self, algo=None, num_epoch=int(1e4), log_interval=500, save_model=False):
        with tqdm(total=num_epoch) as pbar:
            pbar.set_description("Start Offline Training!")

            best_returns = -9999.
            for epoch in range(num_epoch):
                loss_dict = algo.update(self.expert_buffer)

                if not epoch % log_interval:
                    eval_length, eval_returns = self.evaluate(algo.actor, device=algo.device)
                    self.logger.add_scalar('eval env/avg length', eval_length, epoch)
                    self.logger.add_scalar('eval env/avg rewards', eval_returns, epoch)
                    pbar.set_description(("eval returns: {:.4f}".format(eval_returns)))

                    if eval_returns > best_returns and save_model:
                        algo.save(self.logger._model_dir)
                        best_returns = eval_returns

                pbar.update(1)