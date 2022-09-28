from vispy.visuals import TextVisual
from vispy.scene import Widget


class Label(Widget):
    def __init__(self, text: str, rotation: float = 0.0, **kwargs) -> None:
        """Label widget.
        :param text: The label.
        :param rotation: The rotation of the label.
        """

        self._text_visual = TextVisual(
            text=text, rotation=rotation, **kwargs
        )
        self.rotation = rotation
        Widget.__init__(self)
        self.add_subvisual(self._text_visual)
        self._set_pos()

    def on_resize(self, event) -> None:
        """Resize event handler.
        :param event: Inste of Event.
        """

        self._set_pos()

    def _set_pos(self) -> None:
        self._text_visual.pos = (self.rect.left + 4, 4)

    @property
    def text(self):
        """Returns the the label text."""

        return self._text_visual.text

    @text.setter
    def text(self, t):
        """Setter for the label text."""

        self._text_visual.text = t
