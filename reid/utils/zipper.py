# -*- coding: utf-8 -*-
# Time    : 2020/9/15 9:15
# Author  : Yichen Lu

from pathlib import Path
import zipfile
from reid.utils.osutils import mkdir_if_missing


class Zipper(object):
    def __init__(self, root='.', zippath=None):
        self.root = Path(root)
        self.zippath = zippath if zippath else self.root / (str(self.root.absolute().name) + '.zip')
        self.ignores = [str(self.zippath)]
        if (self.root / '.zipignore').exists():
            with open(self.root / '.zipignore', 'r') as file:
                self.ignores.extend([line.strip() for line in file.readlines()])

    def zip(self, ignores=None):
        mkdir_if_missing(Path(self.zippath).parent)
        ignores = list(ignores) + self.ignores if ignores else self.ignores
        with zipfile.ZipFile(self.zippath, 'w') as zfile:
            self.zipdir(self.root, zfile, ignores)

    def zipdir(self, root, zfile, ignores):
        for path in Path(root).iterdir():
            if path.is_file() and not self.is_ignored(ignores, path):
                zfile.write(path)
                print(f"Zip {path} into {self.zippath}.")
            elif path.is_dir() and not self.is_ignored(ignores, path):
                self.zipdir(path, zfile, ignores)

    @staticmethod
    def is_ignored(ignores, path):
        path = Path(path)
        for ignore in ignores:
            if path.match(ignore):
                return True
        return False


if __name__ == '__main__':
    zipper = Zipper('.')
    zipper.zip(['__pycache__'])
    pass


