from dl_norms.batch_norm import test_BatchNorm2d
from dl_norms.group_norm import test_GroupNorm
from dl_norms.instance_norm import test_InstanceNorm2d
from dl_norms.layer_norm import test_LayerNorm


def main():
    test_BatchNorm2d()
    test_InstanceNorm2d()
    test_LayerNorm()
    test_GroupNorm()


if __name__ == '__main__':
    main()
