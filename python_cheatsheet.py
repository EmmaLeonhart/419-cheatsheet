# ============================================================
# PYTHON CHEATSHEET - Quick Reference
# ============================================================

# ------------------------------------------------------------
# 1. VARIABLES & DATA TYPES
# ------------------------------------------------------------
x = 10              # int
y = 3.14            # float
name = "hello"      # str
flag = True          # bool
nothing = None       # NoneType

# Type conversion
int("42")            # str -> int
float("3.14")        # str -> float
str(100)             # int -> str
bool(0)              # int -> bool (False)
list("abc")          # str -> list ['a','b','c']

# ------------------------------------------------------------
# 2. STRINGS
# ------------------------------------------------------------
s = "Hello, World!"
s.upper()            # "HELLO, WORLD!"
s.lower()            # "hello, world!"
s.strip()            # remove leading/trailing whitespace
s.split(", ")        # ["Hello", "World!"]
s.replace("Hello", "Hi")  # "Hi, World!"
s.startswith("Hello")     # True
s.find("World")           # 7 (index), -1 if not found
len(s)                     # 13

# f-strings (formatted strings)
name, age = "Alice", 30
f"Name: {name}, Age: {age}"       # "Name: Alice, Age: 30"
f"{3.14159:.2f}"                   # "3.14"
f"{'hello':>10}"                   # "     hello" (right-align)
f"{'hello':<10}"                   # "hello     " (left-align)
f"{'hello':^10}"                   # "  hello   " (center)

# Multi-line strings
text = """Line 1
Line 2
Line 3"""

# String slicing
s[0]       # 'H'
s[-1]      # '!'
s[0:5]     # 'Hello'
s[::2]     # 'Hlo ol!'  (every 2nd char)
s[::-1]    # '!dlroW ,olleH' (reversed)

# ------------------------------------------------------------
# 3. LISTS
# ------------------------------------------------------------
nums = [1, 2, 3, 4, 5]
nums.append(6)           # [1,2,3,4,5,6]
nums.insert(0, 0)        # [0,1,2,3,4,5,6]
nums.pop()               # removes & returns last item
nums.pop(0)              # removes & returns item at index
nums.remove(3)           # removes first occurrence of 3
nums.extend([7, 8])      # append multiple items
nums.sort()              # sort in place
nums.sort(reverse=True)  # sort descending
nums.reverse()           # reverse in place
nums.index(2)            # index of first occurrence
nums.count(2)            # count occurrences
len(nums)                # length

# Slicing (same as strings)
nums[1:3]    # elements at index 1,2
nums[::2]    # every 2nd element
nums[::-1]   # reversed copy

# List comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
flat = [x for row in [[1,2],[3,4]] for x in row]  # [1,2,3,4]

# Unpacking
a, b, *rest = [1, 2, 3, 4, 5]  # a=1, b=2, rest=[3,4,5]

# ------------------------------------------------------------
# 4. DICTIONARIES
# ------------------------------------------------------------
d = {"name": "Alice", "age": 30}
d["name"]                # "Alice" (KeyError if missing)
d.get("name")            # "Alice" (None if missing)
d.get("x", "default")   # "default" if key missing
d["email"] = "a@b.com"  # add/update key
del d["age"]             # delete key
d.pop("name")            # remove & return value
d.keys()                 # dict_keys([...])
d.values()               # dict_values([...])
d.items()                # dict_items([(k,v), ...])
d.update({"a": 1})       # merge another dict
len(d)                   # number of keys
"name" in d              # True if key exists

# Dict comprehension
squares = {x: x**2 for x in range(5)}
# {0:0, 1:1, 2:4, 3:9, 4:16}

# Merge dicts (Python 3.9+)
merged = {**d, **{"new_key": "val"}}
merged = d | {"new_key": "val"}       # 3.9+

# ------------------------------------------------------------
# 5. SETS & TUPLES
# ------------------------------------------------------------
# Sets (unordered, unique)
s = {1, 2, 3}
s.add(4)
s.remove(2)        # KeyError if missing
s.discard(2)       # no error if missing
s1 = {1, 2, 3}
s2 = {2, 3, 4}
s1 | s2            # union:        {1,2,3,4}
s1 & s2            # intersection: {2,3}
s1 - s2            # difference:   {1}
s1 ^ s2            # symmetric:    {1,4}

# Tuples (immutable)
t = (1, 2, 3)
t[0]               # 1
a, b, c = t        # unpacking
single = (42,)     # single-element tuple (note the comma)

# ------------------------------------------------------------
# 6. CONTROL FLOW
# ------------------------------------------------------------
# If / elif / else
if x > 0:
    print("positive")
elif x == 0:
    print("zero")
else:
    print("negative")

# Ternary
result = "yes" if x > 0 else "no"

# For loop
for item in [1, 2, 3]:
    print(item)

for i in range(5):          # 0,1,2,3,4
    print(i)

for i in range(2, 10, 2):   # 2,4,6,8
    print(i)

for i, val in enumerate(["a", "b", "c"]):  # index + value
    print(i, val)

for k, v in d.items():      # dict iteration
    print(k, v)

# While loop
while x > 0:
    x -= 1

# Loop control
# break     - exit loop
# continue  - skip to next iteration
# else      - runs if loop completes without break

# Match (Python 3.10+)
match command:
    case "quit":
        exit()
    case "hello":
        print("hi")
    case _:
        print("unknown")

# ------------------------------------------------------------
# 7. FUNCTIONS
# ------------------------------------------------------------
def greet(name, greeting="Hello"):
    """Docstring: describe the function."""
    return f"{greeting}, {name}!"

# *args and **kwargs
def func(*args, **kwargs):
    print(args)    # tuple of positional args
    print(kwargs)  # dict of keyword args

func(1, 2, key="value")

# Lambda
double = lambda x: x * 2
add = lambda a, b: a + b

# Map, Filter, Reduce
list(map(lambda x: x*2, [1,2,3]))        # [2,4,6]
list(filter(lambda x: x>2, [1,2,3,4]))   # [3,4]

from functools import reduce
reduce(lambda a, b: a+b, [1,2,3,4])      # 10

# Decorators
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before")
        result = func(*args, **kwargs)
        print("After")
        return result
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

# ------------------------------------------------------------
# 8. CLASSES
# ------------------------------------------------------------
class Animal:
    species_count = 0           # class variable

    def __init__(self, name, sound):
        self.name = name        # instance variable
        self.sound = sound
        Animal.species_count += 1

    def speak(self):
        return f"{self.name} says {self.sound}"

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Animal({self.name!r})"

# Inheritance
class Dog(Animal):
    def __init__(self, name):
        super().__init__(name, "Woof")

    def fetch(self, item):
        return f"{self.name} fetches {item}"

dog = Dog("Rex")
dog.speak()       # "Rex says Woof"
dog.fetch("ball") # "Rex fetches ball"

# Dataclasses (Python 3.7+)
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    label: str = "origin"

p = Point(1.0, 2.0)

# ------------------------------------------------------------
# 9. ERROR HANDLING
# ------------------------------------------------------------
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")
except (TypeError, ValueError):
    print("Type or value error")
except Exception as e:
    print(f"Unexpected: {e}")
else:
    print("No error occurred")
finally:
    print("Always runs")

# Raise exceptions
raise ValueError("Invalid input")

# Custom exception
class MyError(Exception):
    pass

# ------------------------------------------------------------
# 10. FILE I/O
# ------------------------------------------------------------
# Read
with open("file.txt", "r", encoding="utf-8") as f:
    content = f.read()          # entire file as string
    # or
    lines = f.readlines()       # list of lines
    # or
    for line in f:              # line by line (memory efficient)
        print(line.strip())

# Write
with open("file.txt", "w", encoding="utf-8") as f:
    f.write("Hello\n")

# Append
with open("file.txt", "a", encoding="utf-8") as f:
    f.write("More text\n")

# JSON
import json
data = {"key": "value", "nums": [1, 2, 3]}
json_str = json.dumps(data, indent=2)        # dict -> str
parsed = json.loads(json_str)                 # str -> dict

with open("data.json", "w") as f:
    json.dump(data, f, indent=2)              # dict -> file

with open("data.json", "r") as f:
    data = json.load(f)                       # file -> dict

# CSV
import csv
with open("data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row["column_name"])

with open("out.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "age"])
    writer.writerow(["Alice", 30])

# ------------------------------------------------------------
# 11. COMMON MODULES
# ------------------------------------------------------------
import os
os.path.exists("file.txt")       # True/False
os.path.join("dir", "file.txt")  # "dir/file.txt"
os.listdir(".")                  # list directory contents
os.makedirs("a/b/c", exist_ok=True)  # create nested dirs
os.getcwd()                      # current working directory
os.path.basename("/a/b/c.txt")  # "c.txt"
os.path.dirname("/a/b/c.txt")   # "/a/b"
os.path.splitext("file.txt")    # ("file", ".txt")

from pathlib import Path         # modern alternative
p = Path("dir") / "file.txt"
p.exists()
p.read_text()
p.write_text("content")
p.stem                           # "file"
p.suffix                         # ".txt"
p.parent                         # Path("dir")
list(Path(".").glob("*.py"))     # find files

import sys
sys.argv             # command line arguments
sys.exit(1)          # exit with code

import datetime
now = datetime.datetime.now()
today = datetime.date.today()
delta = datetime.timedelta(days=7)
next_week = today + delta

import re
re.search(r"\d+", "abc123")       # <Match '123'>
re.findall(r"\d+", "a1b2c3")      # ['1','2','3']
re.sub(r"\d+", "X", "a1b2")       # "aXbX"
re.split(r"[,;]", "a,b;c")        # ['a','b','c']

from collections import Counter, defaultdict, deque
Counter("abracadabra")              # Counter({'a':5,'b':2,...})
dd = defaultdict(list)
dd["key"].append(1)                 # no KeyError
dq = deque([1, 2, 3])
dq.appendleft(0)                    # [0,1,2,3]

import itertools
itertools.chain([1,2], [3,4])       # 1,2,3,4
itertools.product("AB", "12")       # A1,A2,B1,B2
itertools.combinations("ABCD", 2)   # AB,AC,AD,BC,BD,CD
itertools.permutations("ABC", 2)    # AB,AC,BA,BC,CA,CB

# ------------------------------------------------------------
# 12. COMMON PATTERNS
# ------------------------------------------------------------
# Sorting
sorted([3,1,2])                          # [1,2,3]
sorted(["b","a","c"], reverse=True)      # ['c','b','a']
sorted(users, key=lambda u: u["age"])    # sort by key

# Zip
list(zip([1,2,3], ["a","b","c"]))  # [(1,'a'),(2,'b'),(3,'c')]
dict(zip(keys, values))            # create dict from two lists

# Any / All
any([False, True, False])   # True  (at least one)
all([True, True, True])     # True  (every one)

# Walrus operator (Python 3.8+)
if (n := len(data)) > 10:
    print(f"Too long: {n}")

# Context manager
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    yield
    print(f"Elapsed: {time.time() - start:.2f}s")

with timer():
    do_something()

# Type hints
def add(a: int, b: int) -> int:
    return a + b

from typing import Optional, Union
def find(name: str) -> Optional[str]:
    ...

# ------------------------------------------------------------
# 13. VIRTUAL ENVIRONMENTS & PACKAGES
# ------------------------------------------------------------
# Create venv:       python -m venv myenv
# Activate (Win):    myenv\Scripts\activate
# Activate (Unix):   source myenv/bin/activate
# Install package:   pip install requests
# Requirements:      pip freeze > requirements.txt
# Install from req:  pip install -r requirements.txt

# ------------------------------------------------------------
# 14. USEFUL ONE-LINERS
# ------------------------------------------------------------
# Flatten nested list
flat = [x for sub in nested for x in sub]

# Remove duplicates (preserving order)
unique = list(dict.fromkeys(items))

# Transpose matrix
transposed = list(zip(*matrix))

# Count items
from collections import Counter
counts = Counter(items).most_common(5)

# Chunk a list
chunks = [lst[i:i+n] for i in range(0, len(lst), n)]

# Dictionary from two lists
d = dict(zip(keys, values))

# Invert a dictionary
inv = {v: k for k, v in d.items()}

# Read file into list of stripped lines
lines = Path("file.txt").read_text().splitlines()
