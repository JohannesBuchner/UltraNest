"""
Sphinx extension to add ReadTheDocs-style "Edit on GitHub" links to the
sidebar.

Loosely based on https://github.com/astropy/astropy/pull/347
"""

import os
import warnings


__licence__ = 'BSD (3 clause)'


def get_github_url(app, view, path):
    return (
        'https://github.com/{project}/{view}/{branch}/{src_path}{path}'.format(
        project=app.config.edit_on_github_project,
        view=view,
        branch=app.config.edit_on_github_branch,
        src_path=app.config.edit_on_github_src_path,
        path=path))


def html_page_context(app, pagename, templatename, context, doctree):
    if templatename != 'page.html':
        return

    if not app.config.edit_on_github_project:
        warnings.warn("edit_on_github_project not specified")
        return
    if not doctree:
        warnings.warn("doctree is None")
        return
    path = os.path.relpath(doctree.get('source'), app.builder.srcdir)
    show_url = get_github_url(app, 'blob', path)
    edit_url = get_github_url(app, 'edit', path)

    context['show_on_github_url'] = show_url
    context['edit_on_github_url'] = edit_url

    # make the sphinx_rtd_theme render the "Edit on GitHub" banner in the
    # breadcrumbs bar (see breadcrumbs.html of the theme)
    github_user, _, github_repo = app.config.edit_on_github_project.partition('/')
    context['display_github'] = True
    context['github_user'] = github_user
    context['github_repo'] = github_repo
    context['github_version'] = app.config.edit_on_github_branch
    conf_py_path = app.config.edit_on_github_src_path
    if conf_py_path and not conf_py_path.startswith('/'):
        conf_py_path = '/' + conf_py_path
    context['conf_py_path'] = conf_py_path


def setup(app):
    app.add_config_value('edit_on_github_project', '', True)
    app.add_config_value('edit_on_github_branch', 'master', True)
    app.add_config_value('edit_on_github_src_path', '', True)  # 'eg' "docs/"
    app.connect('html-page-context', html_page_context)
    
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
