#!/bin/sh

# failure
../examples/test5 --aaa asdf -c fdas --fff blah -i one -i two -j huh > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test57.out; then
  exit 0
else
  exit 1
fi

