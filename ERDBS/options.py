import argparse
import template

parser = argparse.ArgumentParser(description ='Efficient Residual Dense Block Search for Image Super-Resolution')

# parser.add_argument('--debug',
#                     action='store_true',
#                     help='Enables debug mode')
# parser.add_argument('--template',
#                     default='.',
#                     help='You can set various templates in option.py')

parser.add_argument("-t", "--task",
                    required=True,
                    choices=['test', 'train', 'evolve'],
                    metavar="Task",
                    help="Task to perform"
                    )
# parser.add_argument("-e", "--evolve",
#                     metavar="Evolve",
#                     help = "Get a model using guided evolutionory method")
# parser.add_argument("-r", "--retrain",
#                     metavar="Retrain",
#                     help = "Get a model using guided evolutionory method")
# parser.add_argument("-t", "--test",
#                     metavar="Test",
#                     help = "Test a model")
parser.add_argument("-s", "--sequence",
                    metavar='Sequence',
                    help="sequence you want to train or test")


parser.add_argument("-d", "--dataset",
                    default='div2k',
                    choices=['div2k', 'b100', 'set5', 'set14', 'urban100'],
                    help="test dataset name")


args = parser.parse_args()
# template.set_template(args)