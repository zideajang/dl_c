CC = gcc
CFLAGS = -I./include
CFLAGS += -lm 
# CFLAGS += -Wall
# CFLAGS += -Wextra
SRC = ./src/main.c ./src/dataloader.c ./src/utils.c ./src/examples.c ./src/mat.c ./src/perceptron.c ./src/nn.c  

OBJ = $(SRC:.c=.o)
TARGET = prog

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

RM=del
TEST_DIR := .\\test
TEST_OBJ_FILES := $(SRC_DIR)/nntest.o
# SLASH := $(shell test -d / && echo / || echo \\)
# $(RM) -f $(OBJ) $(TARGET)
clean:
	@echo $(RM)
	@for file in $(OBJ_FILES); do \
        if [ -f "$(file)" ]; then \
            $(RM) -f "$(file)"; \
        fi; \
    done

# test/mattest.o:mattest.c
# 	$(CC) $(CFLAGS) -c ./test/mattest.c
EXCLUDE := ./src/main.o
NEW_OBJ := $(filter-out $(EXCLUDE), $(OBJ))

MATTESTMATOBJ = $(NEW_OBJ)
NNTESTMATOBJ = $(NEW_OBJ)

MATTESTMATOBJ += ./test/mattest.o
NNTESTMATOBJ += ./test/nntest.o
XORTESTMATOBJ += ./test/xortest.o


testmat:$(MATTESTMATOBJ)
	@echo $(MATTESTMATOBJ)
	$(CC) $(CFLAGS) -o mattest $^ -lm 

testnn:$(NNTESTMATOBJ)
	@echo $(NNTESTMATOBJ)
	$(CC) $(CFLAGS) -o nntest $^ -lm 


testxor:$(XORTESTMATOBJ)
	@echo $(XORTESTMATOBJ)
	$(CC) $(CFLAGS) -o xortest $^ -lm 