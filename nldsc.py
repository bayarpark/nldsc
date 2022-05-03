#!/usr/bin/env python
from src import *


if __name__ == '__main__':
    args = vars(parser.parse_args())
    dispatch(**args)

