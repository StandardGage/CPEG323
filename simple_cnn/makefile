CC = gcc
CFLAGS = -Wall -Wextra -I./lodepng
TARGET = simple_cnn
SRC = main.c simple_cnn.c lodepng/lodepng.c
OBJ = $(SRC:.c=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
