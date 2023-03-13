import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict
import regex
script_list = [
    r'\p{Arabic}', r'\p{Armenian}', r'\p{Bengali}', r'\p{Bopomofo}', r'\p{Braille}',
    r'\p{Buhid}', r'\p{Canadian_Aboriginal}', r'\p{Cherokee}', r'\p{Cyrillic}',
    r'\p{Devanagari}', r'\p{Ethiopic}', r'\p{Georgian}', r'\p{Greek}', r'\p{Gujarati}',
    r'\p{Gurmukhi}', r'\p{Han}', r'\p{Hangul}', r'\p{Hanunoo}', r'\p{Hebrew}', r'\p{Hiragana}',
    r'\p{Inherited}', r'\p{Kannada}', r'\p{Katakana}', r'\p{Khmer}', r'\p{Lao}', r'\p{Latin}',
    r'\p{Limbu}', r'\p{Malayalam}', r'\p{Mongolian}', r'\p{Myanmar}', r'\p{Ogham}', r'\p{Oriya}',
    r'\p{Runic}', r'\p{Sinhala}', r'\p{Syriac}', r'\p{Tagalog}', r'\p{Tagbanwa}',
    r'\p{TaiLe}', r'\p{Tamil}', r'\p{Telugu}', r'\p{Thaana}', r'\p{Thai}', r'\p{Tibetan}',
    r'\p{Yi}', r'\p{Common}'
]
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
"train and test read with shapes : ", train.shape, test.shape
script_occ = pd.DataFrame(
    [regex.sub(r'\\p\{(.+)\}', r'\g<1>', reg) for reg in script_list],
    columns=["script"]
)
script_occ["train"] = [
    train["comment_text"].apply(lambda x: len(regex.findall(reg, x))).sum()
    for reg in script_list
]
script_occ["test"] = [
    test["comment_text"].apply(lambda x: len(regex.findall(reg, x))).sum()
    for reg in script_list
]
script_occ["train_docs"] = [
    (train["comment_text"].apply(lambda x: len(regex.findall(reg, x))) > 0).sum()
    for reg in script_list
]
script_occ["test_docs"] = [
    (test["comment_text"].apply(lambda x: len(regex.findall(reg, x))) > 0).sum()
    for reg in script_list
]
from bokeh.core.properties import value
from bokeh.io import show, output_notebook
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from math import pi

output_notebook()

script_occ.sort_values(by="test", ascending=False, inplace=True)

scripts = list(script_occ.script.values)
dataset = ["test", "train"]

colors = ["#c9d9d3", "#718dbf"]

data = {
    'scripts' : scripts,
    'train': list(np.log1p(script_occ.train.values)),
    'test': list(np.log1p(script_occ.test.values)),
    'real_trn_occ': list(script_occ.train.values),
    'real_sub_occ': list(script_occ.test.values)
}

source = ColumnDataSource(data=data)

hover = HoverTool(tooltips=[
    ("Script", "@scripts"),
    ("Train occurence", "@real_trn_occ"),
    ("Test occurence", "@real_sub_occ"),
])
p = figure(x_range=scripts, plot_height=500, plot_width=850, title="Unicode Script Categories Occurence",
           toolbar_location=None, tools=[hover])

p.vbar_stack(dataset, x='scripts', width=0.9, color=colors, source=source,
             legend=[value(x) for x in dataset])

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.xaxis.major_label_orientation = pi/3
p.outline_line_color = None
p.legend.location = "top_left"
p.legend.orientation = "horizontal"
show(p)
script_occ.sort_values(by="test_docs", ascending=False, inplace=True)

scripts = list(script_occ.script.values)
dataset = ["test", "train"]

colors = ["#c9d9d3", "#718dbf"]

data = {
    'scripts' : scripts,
    'train': list(np.log1p(script_occ.train_docs.values)),
    'test': list(np.log1p(script_occ.test_docs.values)),
    'real_trn_occ': list(script_occ.train_docs.values),
    'real_sub_occ': list(script_occ.test_docs.values)
}

source = ColumnDataSource(data=data)

hover = HoverTool(tooltips=[
    ("Script", "@scripts"),
    ("Number of comments in train", "@real_trn_occ"),
    ("Number of comments in test", "@real_sub_occ"),
])
p = figure(x_range=scripts, plot_height=500, plot_width=850, title="Comments impacted by each Unicode Script Category",
           toolbar_location=None, tools=[hover])

p.vbar_stack(dataset, x='scripts', width=0.9, color=colors, source=source,
             legend=[value(x) for x in dataset])

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.xaxis.major_label_orientation = pi/3
p.outline_line_color = None
p.legend.location = "top_left"
p.legend.orientation = "horizontal"
show(p)
