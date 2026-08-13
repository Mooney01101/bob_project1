
# What does it do?:

Detects the state of the object (example used:bob), whether it's in a hand, midair, or on the ground.

# How does it do that?: 

Using Nanoowl, it detects the object and the hand, with the final stage being determined by a lack of motion by the object.

# Why did I choose this?:

I chose this because it's the simple beginnings of a long-term project that I am developing.

# Preparation:

Before you run the code, please acquire your object (preferably brightly colored/neon). Before running the program, please place your hand holding the object out of frame until you start. As the object may take a few seconds to fully stop after landing, please note that the sequence will not be complete until the object as no motion.

# How to set up:

1) Begin by opening a terminal in VS code and running the following comand:

         jetson-containers run --workdir /opt/nanoowl $(autotag nanoowl)

2) Next, run the following command (ensure you have the full code prepared to paste):

        nano ball_sequence_app.py

3) Afterwards, paste the entire code into the docker using ctrl A, ctrl C, then ctrl V

4) Press ctrl o, enter, then ctrl x

5) Enter the following code

        python3 ball_sequence_app.py

6)  Open a search engine and enter the following by replacing "<YOUR.IP.ADDRESS>" with your IP address

        http://<YOUR.IP.ADDRESS>:7860

    DEMO:[ ](https://drive.google.com/file/d/1aCHXOQX8w8-PRni7bj1iVFFB9ebn1Ced/view?usp=sharing)
    <img width="1375" height="707" alt="image" src="https://github.com/user-attachments/assets/b244d97c-45ba-4716-be31-7c0dcf9f0488" />
    <img width="1358" height="713" alt="image" src="https://github.com/user-attachments/assets/1084c52a-f58a-4d40-b079-1d39774448ed" />
    <img width="1353" height="715" alt="image" src="https://github.com/user-attachments/assets/6925c2c2-3570-4457-b04c-e498c29badd0" />

