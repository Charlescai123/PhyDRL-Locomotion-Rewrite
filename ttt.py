import hydra
from omegaconf import DictConfig, OmegaConf

def train(model: str, n_epoches: int, lr: float, batch_size: int):
    print("Start training...")
    ...

# version_base用于选择Hydra在不同版本下的表现，不是很重要，具体请自行查阅https://hydra.cc/docs/upgrades/version_base/
# config_path表示配置文件所在路径
# config_name表示配置文件文件名，不包含后缀
@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg)) # 打印配置

    # 像属性一样访问配置
    model_name: str = cfg.train_setting.model
    n_epoches: int = cfg.train_setting.n_epoches

    # 像字典一样访问配置
    lr: float = cfg['train_setting']['lr']
    batch_size: int = cfg['train_setting']['batch_size']

    train(model, n_epoches, lr, batch_size) # 用给定配置训练模型

if __name__ == "__main__":
    my_app()


from hydra.utils import instantiate

config = {
    "db": {
        "_target_": "my_module.Database",
        "host": "localhost",
        "port": 5432,
        "username": "myuser",
        "password": "mypassword"
    }
}

db = instantiate(config)
print(db)