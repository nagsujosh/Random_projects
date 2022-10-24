    import turtle
import random

turtle.bgcolor('green')

line = turtle.Turtle()
line.hideturtle()
line.width(5)
line.color('blue')
line.penup()
line.setpos(-250, 300)
line.pendown()
line.setheading(0)
line.forward(500)
line.penup()
line.setpos(-250, -310)
line.down()
line.setheading(0)
line.forward(500)

gamer1 = turtle.Turtle()
gamer2 = turtle.Turtle()

gamer1.speed(50)
gamer2.speed(50)

gamer1.pensize(3)
gamer2.pensize(3)

gamer1.color('white')
gamer2.color('white')

gamer1.shapesize(1, 1, 1)
gamer2.shapesize(1, 1, 1)

gamer1.penup()
gamer1.setpos(-200, -300)
gamer1.pendown()
gamer1.setheading(90)

gamer2.penup()
gamer2.setpos(200, -300)
gamer2.pendown()
gamer2.setheading(90)

line.speed(3)
line.color('yellow')
line.setpos(-250, -310)

count1 = 0
count2 = 0
line.speed(-1)
for i in range(1000):
    num1 = random.choice((1, 1.5, 1.75, 1.25, 0.25, 0, 0.5, 0.75)) + random.random()
    num2 = random.choice((1, 1.5, 1.75, 1.25, 0.25, 0, 0.5, 0.75)) + random.random()

    count1 += num1
    count2 += num2

    gamer1.forward(num1)
    gamer2.forward(num2)

    if count1 >= 600:
        print('Gamer1 Win.Congrats!')
        print(f'Gamer2 lose for {int(600-count2)} m.')
        line.penup()
        line.setpos(-250, 300)
        line.pendown()
        line.setheading(0)
        line.forward(500)
        gamer1.color('red')
        gamer1.shapesize(2, 2, 2)
        with open('game_result.txt', 'a') as rf:
            rf.write('Gamer1 win.\n')
        break

    elif count2 >= 600:
        print('Gamer2 Win.Congrats.')
        print(f'Gamer1 lose for {int(600-count1)} m.')
        line.penup()
        line.setpos(-250, 300)
        line.pendown()
        line.setheading(0)
        line.forward(500)
        gamer2.color('red')
        gamer2.shapesize(2, 2, 2)
        with open('game_result.txt', 'a') as rf:
            rf.write('Gamer2 win.\n')
        break

turtle.done()
