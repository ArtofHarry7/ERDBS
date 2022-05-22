from logging import raiseExceptions
from options import args
import test
import retrain
import evolve

def main():
    if args.task == 'test':
        if not args.sequence:
            raiseExceptions('Are bhai blocks ka sequence to daal')
        test.run(args.dataset, list(args.sequence))
    elif args.task == 'train':
        if not args.sequence:
            raiseExceptions('Are bhai blocks ka sequence to daal')
        retrain.run(list(args.sequence))
    elif args.task == 'evolve':
        evolve.run()

if __name__=="__main__":
    main()