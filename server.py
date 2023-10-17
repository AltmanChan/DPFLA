from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.fl_core import FL
from federated_learning.my_dict import get_cifar10_labels_dict, get_mnist_labels_dict


def generate_log_file(args):
    savepath = './logs_3/' + args.get_dataset_name() + '_' + args.get_model_name() + '_' + args.get_dd_type() + '_' + args.get_attack_type() + '_' + args.get_rule() + '_' + str(
        args.get_malicious_rate()) + '_' + str(args.get_local_epochs()) + '.log'
    return savepath


def run_exp(num_workers, frac_workers, attack_type, rule, replace_method, dataset, malicious_rate,
            malicious_behavior_rate, global_round, local_epoch, untarget):
    args = Arguments(logger)
    args.set_num_workers(num_workers)
    args.set_frac_workers(frac_workers)
    args.set_rule(rule)
    args.set_source_class(replace_method['source_class'])
    args.set_target_class(replace_method['target_class'])
    args.set_attack_type(attack_type)
    args.set_malicious_rate(malicious_rate)
    args.set_malicious_behavior_rate(malicious_behavior_rate)
    args.set_global_rounds(global_round)
    args.set_local_epochs(local_epoch)

    args.set_dataset_name(dataset['dataset_name'])
    args.set_model_name(dataset['model_name'])
    if dataset['dataset_name'] == 'MNIST':
        args.set_labels_dict(get_mnist_labels_dict())
    elif dataset['dataset_name'] == 'CIFAR10':
        args.set_labels_dict(get_cifar10_labels_dict())
    else:
        raise Exception('Undefined dataset!!!')

    log_files = generate_log_file(args)
    # Initialize logger
    handler = logger.add(log_files, enqueue=True)

    args.log()

    flEnv = FL(dataset_name=args.get_dataset_name(), model_name=args.get_model_name(), dd_type=args.get_dd_type(),
               num_clients=args.get_num_workers(), frac_clients=args.get_frac_workers(), seed=args.get_seed(),
               test_batch_size=args.get_test_batch_size(), criterion=args.get_loss_function(),
               global_rounds=args.get_global_rounds(),
               local_epochs=args.get_local_epochs(), local_bs=args.get_batch_size(), local_lr=args.get_lr(),
               local_momentum=args.get_momentum(), labels_dict=args.get_labels_dict(), device=args.get_device(),
               attackers_ratio=args.get_malicious_rate(),
               class_per_client=args.get_class_per_workers(), samples_per_class=args.get_samples_per_class(),
               rate_unbalance=args.get_rate_unbalance(), alpha=args.get_alpha(), source_class=args.get_source_class())

    flEnv.run_experiment(attack_type=args.get_attack_type(), malicious_behavior_rate=args.get_malicious_behavior_rate(),
                         source_class=args.get_source_class(), target_class=args.get_target_class(),
                         rule=args.get_rule(), resume=False, model_name=args.get_model_name(), untarget=untarget)

    logger.remove(handler)
