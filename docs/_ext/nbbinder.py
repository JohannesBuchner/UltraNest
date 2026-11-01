"""
Sphinx extension to add a "Run this notebook in Binder" link at the top
(before the title) of Jupyter notebook pages rendered by nbsphinx.
"""

import os
import warnings
from urllib.parse import quote

from docutils import nodes


def get_binder_url(app, path):
    return (
        'https://{org}/v2/gh/{project}/{branch}?filepath={path}'.format(
        org=app.config.binder_org,
        project=app.config.binder_project,
        branch=app.config.binder_branch,
        path=quote(app.config.binder_src_path + path)))


def doctree_read(app, doctree):
    source = doctree.get('source')
    if not source or not source.endswith('.ipynb'):
        return
    if not app.config.binder_project:
        warnings.warn("binder_project not specified")
        return
    path = os.path.relpath(source, app.srcdir)
    url = get_binder_url(app, path)

    para = nodes.paragraph(classes=['binder-banner'])
    ref = nodes.reference(refuri=url, internal=False)
    ref.append(nodes.Text('Run this notebook in Binder'))
    para.append(ref)
    # insert before the title (first section) of the notebook
    doctree.insert(0, para)


def setup(app):
    app.add_config_value('binder_org', 'mybinder.org', 'env')
    app.add_config_value('binder_project', '', 'env')  # 'user/repo'
    app.add_config_value('binder_branch', 'master', 'env')
    app.add_config_value('binder_src_path', '', 'env')  # e.g. 'docs/'
    app.add_css_file('binder.css')
    app.connect('doctree-read', doctree_read)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
