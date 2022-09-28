import tkinter as tk


class PredictionFrame(tk.Frame):

    def __init__(self, master, toggle_predict, set_confidence):
        super().__init__(master)
        self._toggle_predict = toggle_predict

        tk.Label(self, text="Confidence").pack()
        self.confidence_slider = tk.Scale(self, from_=0, to=1, resolution=0.01,
                                          command=set_confidence)
        self.confidence_slider.set(0.5)
        self.confidence_slider.pack()

        self._predict_button = tk.Button(self, anchor="e", text="Predict",
                                         command=self.toggle_predict)
        self._predict_button.pack(side=tk.BOTTOM)

    def toggle_predict(self) -> None:
        if self._predict_button.config('relief')[-1] == 'sunken':
            self._predict_button.config(relief="raised")
            self._toggle_predict(False)
        else:
            self._predict_button.config(relief="sunken")
            self._toggle_predict(True)
