from glumpy import app

window = app.Window()

@window.event
def on_draw(dt):
    window.clear()

app.run()