from typing import Literal

from PySide6.QtCore import Qt, QEvent, QObject, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPixmap, QTransform
from PySide6.QtWidgets import (
    QWidget, QDockWidget, QLabel, QFrame, QPushButton, 
    QDockWidget, QStackedWidget)

import pix2pix_graphical.ui.utils.common as utils

class SettingsDock(QObject):
    def __init__(self, mainwidget: QWidget):
        super().__init__()
        self.mainwidget = mainwidget
        self.find = self.mainwidget.findChild
        self._get_widgets()

        self.button_names = [
            'experiment', 'dataset', 'training', 'testing', 'review', 'utils', 'settings']
        self.button_icons = [
            'flask', 'database', 'barbell', 'test-tube', 'clipboard', 'wrench', 'gear']

    ### INIT HELPERS ###

    def _get_widgets(self) -> None:
        self.dock = self.find(QDockWidget, 'settings_dock')
        self.main_icon = self.find(QLabel, 'settings_dock_main_icon')
        self.section_header = self.find(QLabel, 'settings_section_header')
        self.section_icon = self.find(QLabel, 'settings_section_header_icon')
        self.btns_frame = self.find(QFrame, 'settings_btns_frame')
        self.title_tag = self.find(QLabel, 'title_tag')
        self.creator_tag = self.find(QLabel, 'creator_tag')

    ### CALLBACKS ###

    def _set_section_icons(self, index: int=0) -> None:
        file = utils.get_icon_file(f'{self.button_icons[index]}.svg')
        transform = QTransform()
        pixmap = QPixmap(file.as_posix()).transformed(transform)
        self.section_icon.setPixmap(pixmap)

    def _change_page(self, index: int) -> None:
        self.section_header.setText(self.button_names[index].capitalize())
        self._set_section_icons(index=index)

        for name in self.button_names:
            button = self.find(QPushButton, f'{name}_settings_btn')
            if not name == self.button_names[index]:
                button.setChecked(False)
            else: button.setChecked(True)
        stack = self.find(QStackedWidget, 'settings_pages')
        try: stack.setCurrentIndex(index)
        except: pass

    def _set_main_icon(self, size: tuple[int]) -> None:
        main_icon_pixmap = QPixmap(utils.get_icon_file('drop.svg')).scaled(
            size[0], size[1],  Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation)
        self.main_icon.setPixmap(main_icon_pixmap)

    def _swap_btn_label_visibility(
            self, 
            mode: Literal['show', 'hide']
        ) -> None:
        if mode == 'show':
            for i, name in enumerate(self.button_names):
                label = self.find(QPushButton, f'{name}_settings_label')
                label.setHidden(False)
        elif mode == 'hide': 
            for i, name in enumerate(self.button_names):
                label = self.find(QPushButton, f'{name}_settings_label')
                label.setHidden(True)
    
    def _init_navbar_expansion(self) -> None:
        self.btns_frame.installEventFilter(self)

    def _animate_nav_panel(self, expand: bool):
        width = 180 if expand else 65
        self.nav_anim = QPropertyAnimation(self.btns_frame, b'maximumWidth')
        self.nav_anim.setDuration(300)
        self.nav_anim.setStartValue(self.btns_frame.width())
        self.nav_anim.setEndValue(width)
        self.nav_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.nav_anim.start()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched == getattr(self, 'btns_frame', None):
            if event.type() == QEvent.Enter:
                self._animate_nav_panel(expand=True)
                self._swap_btn_label_visibility('show')
                self.title_tag.setText('NectarGAN Visual Trainer')
                self.creator_tag.setText('Created by Zachary Bork')
            elif event.type() == QEvent.Leave:
                self._animate_nav_panel(expand=False)
                self._swap_btn_label_visibility('hide')
                self.title_tag.setText('')
                self.creator_tag.setText('')
        return super().eventFilter(watched, event)

    def init(self) -> None:
        self.dock.setTitleBarWidget(QWidget(None))
        self.title_tag.setText('')
        self.creator_tag.setText('')
        self._set_main_icon(size=(50, 50))

        self.btns_frame.setMouseTracking(True)
        self._init_navbar_expansion()
        self._set_section_icons()

        for i, name in enumerate(self.button_names):
            button = self.find(QPushButton, f'{name}_settings_btn')
            button.clicked.connect(lambda x, i=i : self._change_page(i))
            utils.set_button_icon(f'{self.button_icons[i]}.svg', button)

            label = self.find(QPushButton, f'{name}_settings_label')
            label.clicked.connect(lambda x, i=i : self._change_page(i))
            label.setHidden(True)

