from docstr_md.python import PySoup, compile_md
from docstr_md.src_href import Github

import os

src_href = Github('https://github.com/dsbowen/fast-automl/blob/master')
filenames = [
    'automl',
    'baseline',
    'cv_estimators',
    'ensemble',
    'linear_model',
    'metrics',
    'test',
    'utils'
]
for filename in filenames:
    path = os.path.join('fast_automl', filename+'.py')
    soup = PySoup(path=path, parser='sklearn', src_href=src_href)
    outfile = os.path.join('docs_md', 'api', filename+'.md')
    # outfile = 'docs_md/'+filename+'.md'
    compile_md(soup, compiler='sklearn', outfile=outfile)