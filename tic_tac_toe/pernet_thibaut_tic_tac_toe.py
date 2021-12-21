import os
os.system("cls")

class Board():
    def __init__(self):
        self.cells = [" "," "," "," "," "," "," "," "," "," "]

    def display(self):
        print(" %s | %s | %s " %(self.cells[7], self.cells[8], self.cells[9]))
        print("-----------")
        print(" %s | %s | %s " %(self.cells[4], self.cells[5], self.cells[6]))
        print("-----------")
        print(" %s | %s | %s " %(self.cells[1], self.cells[2], self.cells[3]))

    def update_cell(self, cell_number, player):
        if self.cells[cell_number] == " ":
            self.cells[cell_number] = player

    def is_winner(self, player):
        if self.cells[7] == player and self.cells[8] == player and self.cells[9] == player:
            return True
        if self.cells[4] == player and self.cells[5] == player and self.cells[6] == player:
            return True
        if self.cells[1] == player and self.cells[2] == player and self.cells[3] == player:
            return True
        if self.cells[7] == player and self.cells[4] == player and self.cells[1] == player:
            return True
        if self.cells[8] == player and self.cells[5] == player and self.cells[2] == player:
            return True
        if self.cells[9] == player and self.cells[6] == player and self.cells[3] == player:
            return True
        if self.cells[7] == player and self.cells[5] == player and self.cells[3] == player:
            return True
        if self.cells[1] == player and self.cells[5] == player and self.cells[9] == player:
            return True 
        return False
    
    def is_draw(self):
        used_cells = 0
        for cell in self.cells:
            if cell != " ":
                used_cells += 1
        if used_cells == 9:
            return True
        else:
            return False

    def board_reset(self):
        self.cells = [" "," "," "," "," "," "," "," "," "," "]

board = Board()

def print_header():
    print("Welcome players!")

def refresh_screen():
    # Clear the screen
    os.system("cls")

    # Print the header
    print_header()
    
    # Show the board
    board.display()

while True:
    refresh_screen()

    # Get X input
    x_choice = int(input("\nX) Please choose position 1 - 9. > "))

    # Update Board
    board.update_cell(x_choice, "X")

    #Refresh screen
    refresh_screen()

    # Check for X win
    if board.is_winner("X"):
        print ("\nX wins!\n")
        play_again = input("Would you like to play again? (Y/N)> ").upper()
        if play_again == "Y":
            board.board_reset()
            continue
        else:
            break

    # Check for draw
    if board.is_draw():
        print ("\nDraw!\n")
        play_again = input("Would you like to play again? (Y/N)> ").upper()
        if play_again == "Y":
            board.board_reset()
            continue
        else:
            break

    # Get O input
    o_choice = int(input("\nO) Please choose position 1 - 9. > "))

    # Update Board
    board.update_cell(o_choice, "O")

    #Refresh screen
    refresh_screen()

    # Check for O win
    if board.is_winner("O"):
        print ("\nO wins!\n")
        play_again = input("Would you like to play again? (Y/N)> ").upper()
        if play_again == "Y":
            board.board_reset()
            continue
        else:
            break

    # Check for draw
    if board.is_draw():
        print ("\nDraw!\n")
        play_again = input("Would you like to play again? (Y/N)> ").upper()
        if play_again == "Y":
            board.board_reset()
            continue
        else:
            break