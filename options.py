import argparse
import template

parser = argparse.ArgumentParser(description ='Efficient Residual Dense Block Search for Image Super-Resolution')

parser.add_argument("-t", "--task",
                    required=True,
                    choices=['test', 'train', 'evolve'],
                    metavar="Task",
                    help="Task to perform")


parser.add_argument("-s", "--sequence",
                    metavar='Sequence',
                    help="sequence you want to train or test")


parser.add_argument("-d", "--dataset",
                    default='div2k',
                    choices=['div2k', 'b100', 'set5', 'set14', 'urban100'],
                    help="test dataset name")


args = parser.parse_args()
# template.set_template(args)