# ML-visualizer

![Travis (.com)](https://img.shields.io/travis/com/eqtstv/ML-visualizer?style=flat-square)
![GitHub last commit](https://img.shields.io/github/last-commit/eqtstv/ML-visualizer?style=flat-square)
![Lines of code](https://img.shields.io/tokei/lines/github/eqtstv/ML-visualizer?style=flat-square)
![Lines of code](https://img.shields.io/badge/code%20style-black-black?style=flat-square)

![Tests](https://github.com/eqtstv/ML-visualizer/workflows/Tests/badge.svg?style=flat-square)
![CodeQL](https://github.com/eqtstv/ML-visualizer/workflows/CodeQL/badge.svg)
![Flake8-black Linter](https://github.com/eqtstv/ML-visualizer/workflows/Flake8-black%20Linter/badge.svg)
[![codecov](https://codecov.io/gh/eqtstv/ML-visualizer/branch/main/graph/badge.svg?token=OZF0KBA6C8)](https://codecov.io/gh/eqtstv/ML-visualizer)

![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue?style=flat-square)
![Flask](https://img.shields.io/badge/flask-1.1.2-blue?style=flat-square)
![Dash](https://img.shields.io/badge/plotly-dash-e4f5f2?style=flat-square)
![Tensorflow](https://img.shields.io/badge/tensorflow-2.4.0-orange?style=flat-square)

Application for visualizing machine learning process parameters

https://live-ml-visualizer.herokuapp.com

![live](https://user-images.githubusercontent.com/38236287/102703099-05bae380-426b-11eb-9960-a73e386bdca1.gif)

---

### User manual

1. Register and login to the server: https://live-ml-visualizer.herokuapp.com

2. Configure the client:

- Download and place client folder `mlvisualizer` next to your ML model.

- Import BatchTracker instance to your code (1) and add it to the list of callbacks (2).

```python
from mlvisualizer.callback import BatchTracker              (1)
...

model = create_model()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=x_train,
          y=y_train,
          epochs=5,
          validation_data=(x_test, y_test),
          callbacks=[BatchTracker()])                       (2)
```

3. Run your ML model file.

```console
> python example_model.py
```

4. Follow instructions appearing in the console:

   - Authorize connection with credentials used on the server
   - Select project name, or create new project
   - After creating new project, it will apprear in your profile page

5. Click selected project card from your profile page. It will automatically transfer you to the dashboard.

6. Start training your model, and track its progress in the dashboard.
