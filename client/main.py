import argparse
from bots import BotLevel1, BotLevel2, BotLevel3


def main(bot_level, iterations, max_open_streams):
    if bot_level == 1:
        bot = BotLevel1(iterations, max_open_streams)
    elif bot_level == 2:
        bot = BotLevel2(iterations, max_open_streams)
    elif bot_level == 3:
        bot = BotLevel3(iterations, max_open_streams)
    else:
        raise RuntimeError("Invalid Bot!")

    bot.connect()
    bot.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pi HTTP server')
    parser.add_argument('-b', '--bot', type=int, help='Bot Level', choices=[1, 2, 3], default=3)
    parser.add_argument('-i', '--iterations', type=int, help='Number of iterations', default=40)
    parser.add_argument('-ms', '--max_streams', type=int, help='Max of open concurrent streams', default=5)
    args = parser.parse_args()

    iterations = args.iterations
    max_open_streams = args.max_streams
    bot_level = args.bot
    main(bot_level, iterations, max_open_streams)
