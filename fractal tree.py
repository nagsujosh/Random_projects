import turtle

def draw_branch(t, branch_length):
    if branch_length > 2:
        t.forward(branch_length)
        t.right(20)
        draw_branch(t, branch_length - 10)
        t.left(40)
        draw_branch(t, branch_length - 10)
        t.right(20)
        t.backward(branch_length)


def draw_fractal_tree(t, branch_length):
    t.speed(10)
    t.left(90)
    t.penup()
    t.backward(100)
    t.pendown()
    draw_branch(t, branch_length)


t = turtle.Turtle()
draw_fractal_tree(t, 80)
turtle.done()
