import argparse

from draw.run import run

def parse_command_line():
    parser = argparse.ArgumentParser('Detecting and Recognizing a Wide Range of Yu-Gi-Oh! Cards', add_help=True)

    parser.add_argument('--model_size', default=0, type=int,
                        help="The size of the model to use, 0 for the standard one and 1 for the large one (default: 0).")

    parser.add_argument("--deck_list", default="", type=str,
                        help="[Optional] Path or link to the deck list.")

    parser.add_argument("--minimum_out_of_screen_time", default=25, type=int,
                        help="Minimum time before a card is allow to be displayed again in seconds (default: 25s).")

    parser.add_argument('--minimum_screen_time', default=6, type=int,
                        help="Minimum time a card is displayed in seconds (default: 6s).")
    parser.add_argument('--confidence_threshold', default=5, type=int,
                        help="Confidence threshold a detection can have to be displayed in % (default: 5%).")

    parser.add_argument('--channel', default=1, type=int,
                        help="Channel of the display source in the plugin (default: 1).")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_command_line()
    run(
        model_size=args.model_size,
        deck_list=args.deck_list,
        minimum_out_of_screen_time=args.minimum_out_of_screen_time,
        minimum_screen_time=args.minimum_screen_time,
        confidence_threshold=args.confidence_threshold,
        channel=args.channel
    )
