CC = g++

CFLAGS = -W -Wall -Wextra -g2 -O2

INCLUDES = -I./include

LFLAGS =

LIBS =

SRC_PATH = src/

SRCS_NAME = main.cpp my_ai.cpp

MAIN = my_ai

SRCS = $(patsubst %, $(SRC_PATH)%, $(SRCS_NAME))

OBJS = $(SRCS:.cpp=.o)

all : $(MAIN)

$(MAIN) : $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

%.o : %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean :
	$(RM) $(OBJS) *~

fclean : clean
	$(RM) $(MAIN)

re : fclean all
