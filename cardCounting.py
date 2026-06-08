import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm

class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value
        
    def get_value(self):
        if self.value in ['J', 'Q', 'K']:
            return 10
        elif self.value == 'A':
            return 11  # Ace value will be adjusted in Hand class
        else:
            return int(self.value)
    
    def __repr__(self):
        return f"{self.value}{self.suit}"

class Deck:
    def __init__(self, num_decks=8):
        self.num_decks = num_decks
        self.cards = []
        self.dealt_cards = []
        self.reset()
        
    def reset(self):
        self.cards = []
        self.dealt_cards = []
        suits = ['H', 'D', 'C', 'S']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        
        for _ in range(self.num_decks):
            for suit in suits:
                for value in values:
                    self.cards.append(Card(suit, value))
        
        random.shuffle(self.cards)
    
    def deal(self):
        if not self.cards:
            return None
        
        card = self.cards.pop(0)
        self.dealt_cards.append(card)
        return card
    
    def cards_remaining(self):
        return len(self.cards)
    
    def needs_shuffle(self):
        # Reshuffle after 350 cards have been dealt (84% penetration on
        # an 8-deck shoe). 72% pen leaves few high counts on the table;
        # 84% is a realistic "deep" penetration for hand-shuffled games
        # and lets counters express more of their edge.
        return len(self.dealt_cards) >= 350
    
    def get_card_count(self):
        # Returns dictionary with count of each card value left in the deck
        count = defaultdict(int)
        for card in self.cards:
            count[card.value] += 1
        return count

class Hand:
    def __init__(self):
        self.cards = []
        self.value = 0
        self.aces = 0
        self.is_soft = False
        self.is_blackjack = False
        self.can_split = False
        self.is_splitted = False
        # True if this hand was created (or had its companion created) by
        # splitting a pair of Aces. Casinos universally restrict split aces
        # to exactly one card -- no hit, no double, no re-split. The env
        # enforces this by forcing STAND on any action against such a hand.
        self.is_split_aces = False
        # True if the player surrendered this hand. They forfeit half the
        # bet and play ends for this hand.
        self.surrendered = False
        
    def add_card(self, card):
        self.cards.append(card)
        self._calculate_value()
        self._check_split()
        
    def _calculate_value(self):
        self.value = 0
        self.aces = 0

        for card in self.cards:
            if card.value == 'A':
                self.aces += 1
            self.value += card.get_value()

        # Downgrade aces from 11 to 1 as needed to avoid busting.
        while self.aces > 0 and self.value > 21:
            self.value -= 10
            self.aces -= 1

        # A hand is "soft" if there is still an Ace counted as 11.
        self.is_soft = self.aces > 0

        # Natural blackjack: A-T or T-A in the first two cards of an
        # un-split hand. Recompute every time so that adding a third card
        # (e.g. via DOUBLE on a natural 21) un-sets the flag -- doubling
        # a blackjack must NOT keep paying 3:2.
        self.is_blackjack = (
            len(self.cards) == 2
            and self.value == 21
            and not self.is_splitted
        )
    
    def _check_split(self):
        if len(self.cards) == 2 and self.cards[0].value == self.cards[1].value:
            self.can_split = True
        else:
            self.can_split = False
    
    def is_busted(self):
        return self.value > 21
    
    def is_pair(self):
        return len(self.cards) == 2 and self.cards[0].value == self.cards[1].value
    
    def get_first_card(self):
        if self.cards:
            return self.cards[0]
        return None
    
    def __repr__(self):
        hand_str = ", ".join(str(card) for card in self.cards)
        return f"{hand_str} ({self.value})"

class CardCounter:
    def __init__(self, strategy='hi-lo'):
        self.strategy = strategy
        self.running_count = 0
        self.true_count = 0
        self.decks_remaining = 8.0
        
        # Define count values for different strategies
        if strategy == 'hi-lo':
            self.count_values = {
                '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
                '7': 0, '8': 0, '9': 0,
                '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
            }
        elif strategy == 'hi-opt-1':
            self.count_values = {
                '2': 0, '3': 1, '4': 1, '5': 1, '6': 1,
                '7': 0, '8': 0, '9': 0,
                '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': 0
            }
        elif strategy == 'hi-opt-2':
            self.count_values = {
                '2': 1, '3': 1, '4': 2, '5': 2, '6': 1,
                '7': 1, '8': 0, '9': 0,
                '10': -2, 'J': -2, 'Q': -2, 'K': -2, 'A': 0
            }
        elif strategy == 'omega-2':
            self.count_values = {
                '2': 1, '3': 1, '4': 2, '5': 2, '6': 2,
                '7': 1, '8': 0, '9': -1,
                '10': -2, 'J': -2, 'Q': -2, 'K': -2, 'A': 0
            }
        else:  # Default to Hi-Lo
            self.count_values = {
                '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
                '7': 0, '8': 0, '9': 0,
                '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
            }
    
    def update_count(self, card):
        self.running_count += self.count_values[card.value]
        self.true_count = self.running_count / self.decks_remaining
    
    def update_decks_remaining(self, cards_remaining):
        self.decks_remaining = cards_remaining / 52
    
    def reset(self):
        self.running_count = 0
        self.true_count = 0
        self.decks_remaining = 8.0
    
    def get_bet_amount(self, min_bet=50, max_bet=5000):
        """
        Determine bet amount based on true count
        Higher counts mean higher bets
        """
        if self.true_count <= 1:
            return min_bet
        elif self.true_count >= 6:
            return max_bet
        else:
            # Linearly scale bet with true count between min and max bet
            bet_multiplier = (self.true_count - 1) / 5  # Scale from 0 to 1
            bet = min_bet + bet_multiplier * (max_bet - min_bet)
            # Round to nearest $50
            return min(int(round(bet / 50) * 50), max_bet)
    
    def __repr__(self):
        return f"Running Count: {self.running_count}, True Count: {self.true_count:.2f}, Decks Remaining: {self.decks_remaining:.2f}"

class BlackjackEnv:
    def __init__(self, num_decks=8, min_bet=50, max_bet=5000):
        self.deck = Deck(num_decks)
        self.counter = CardCounter(strategy='hi-lo')
        self.player_hands = []
        self.dealer_hand = None
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.bets = []
        self.current_hand_index = 0
        self.game_over = False
        self.bankroll = 10000  # Start with $10,000 bankroll
        self.num_hands_played = 0
        self.results_history = []
        
    def reset(self):
        """
        Reset the game to start a new hand
        """
        # Check if deck needs shuffling
        if self.deck.needs_shuffle():
            self.deck.reset()
            self.counter.reset()

        self.player_hands = [Hand()]
        self.dealer_hand = Hand()
        self.bets = [self.counter.get_bet_amount(self.min_bet, self.max_bet)]
        self.current_hand_index = 0
        self.game_over = False
        # US peek tracking. dealer_hole_dealt is True if reset already dealt
        # the hole card (i.e., upcard was 10/A so we peeked). dealer_pre_bj
        # is True if the peek revealed a natural; the first step() call
        # will then resolve the round before any player action.
        self.dealer_hole_dealt = False
        self.dealer_pre_bj = False

        # Deal initial cards
        for _ in range(2):
            for hand in self.player_hands:
                card = self.deck.deal()
                hand.add_card(card)
                self.counter.update_count(card)

            if _ == 0:  # Only deal one card to dealer initially (the visible card)
                card = self.deck.deal()
                self.dealer_hand.add_card(card)
                self.counter.update_count(card)

        # Update decks remaining
        self.counter.update_decks_remaining(self.deck.cards_remaining())

        # ---- US peek for blackjack ----
        # If the dealer's upcard is a ten-valued card or an Ace, deal the
        # hole face-down and check whether the dealer has a natural. If
        # the dealer has BJ, the round ends BEFORE the player acts (so the
        # player can't lose double/split bets to a dealer natural). The
        # hole card is removed from the deck but only added to the running
        # count once it's revealed (immediately on BJ; otherwise when the
        # dealer plays out at end of round).
        upcard_val = self.dealer_hand.cards[0].get_value()
        if upcard_val in (10, 11):
            hole = self.deck.deal()
            self.dealer_hand.add_card(hole)
            self.dealer_hole_dealt = True
            self.counter.update_decks_remaining(self.deck.cards_remaining())
            if self.dealer_hand.is_blackjack:
                self.counter.update_count(hole)
                self.dealer_pre_bj = True
        
        # Get initial state
        return self._get_state()
    
    def _get_state(self):
        """
        Return current state of the game
        """
        if self.current_hand_index < len(self.player_hands):
            current_hand = self.player_hands[self.current_hand_index]
            
            # Dealer's up card value
            dealer_card = self.dealer_hand.get_first_card()
            dealer_up_card = dealer_card.get_value() if dealer_card else 0
            
            # Player's current hand value
            player_value = current_hand.value
            is_soft = current_hand.is_soft
            can_split = current_hand.can_split
            
            # Card counting information
            running_count = self.counter.running_count
            true_count = self.counter.true_count
            decks_remaining = self.counter.decks_remaining
            
            # Remaining card distribution
            card_distribution = self.deck.get_card_count()
            
            return {
                'dealer_up_card': dealer_up_card,
                'player_hand': current_hand,
                'player_value': player_value,
                'is_soft': is_soft,
                'can_split': can_split,
                'running_count': running_count,
                'true_count': true_count,
                'decks_remaining': decks_remaining,
                'card_distribution': card_distribution,
                'bankroll': self.bankroll,
                'bet': self.bets[self.current_hand_index]
            }
        else:
            return None
    
    def step(self, action):
        """
        Take action; return (state, reward, done).
        Actions: 0 = hit, 1 = stand, 2 = double, 3 = split, 4 = surrender

        Accounting rule: NO intermediate bankroll/reward updates. We only
        compute the bankroll change and per-hand reward at the very end of
        the play (when current_hand_index passes the last player hand),
        via _calculate_rewards. This avoids the double-debit-on-bust bug
        that the old logic had with split hands.
        """
        if self.game_over:
            return self._get_state(), 0, True

        # US peek caught a dealer natural during reset. End the round on
        # the very first step call -- the player doesn't get to act.
        if self.dealer_pre_bj:
            self.game_over = True
            reward = self._calculate_rewards()
            self.bankroll += reward
            self.num_hands_played += 1
            return self._get_state(), reward, True

        if self.current_hand_index >= len(self.player_hands):
            return self._get_state(), 0, True

        current_hand = self.player_hands[self.current_hand_index]
        advance_to_next = False  # set True if we're done with the current hand

        # Casino rule: split aces get exactly one card. No hit, no double,
        # no re-split. Force STAND on any action against a split-aces hand.
        if current_hand.is_split_aces:
            action = 1

        if action == 0:  # Hit
            card = self.deck.deal()
            current_hand.add_card(card)
            self.counter.update_count(card)
            if current_hand.is_busted():
                advance_to_next = True

        elif action == 1:  # Stand
            advance_to_next = True

        elif action == 2:  # Double
            if len(current_hand.cards) == 2:
                self.bets[self.current_hand_index] *= 2
                card = self.deck.deal()
                current_hand.add_card(card)
                self.counter.update_count(card)
                advance_to_next = True
            else:
                # Can't double after hitting -- fall back to a plain hit.
                card = self.deck.deal()
                current_hand.add_card(card)
                self.counter.update_count(card)
                if current_hand.is_busted():
                    advance_to_next = True

        elif action == 3:  # Split
            if (current_hand.can_split
                    and len(current_hand.cards) == 2
                    and len(self.player_hands) < 4):  # cap at 4 hands
                current_bet = self.bets[self.current_hand_index]
                # If splitting Aces, propagate the no-further-play rule onto
                # both resulting hands.
                is_ace_split = current_hand.cards[0].value == 'A'

                new_hand = Hand()
                new_hand.is_splitted = True
                # The current hand is also now a split hand -- this matters
                # for blackjack detection. A,10 after a split is just 21,
                # NOT a natural blackjack.
                current_hand.is_splitted = True
                if is_ace_split:
                    current_hand.is_split_aces = True
                    new_hand.is_split_aces = True

                card = current_hand.cards.pop()
                new_hand.add_card(card)

                current_hand._calculate_value()
                current_hand._check_split()

                # Deal one card to each split hand.
                card = self.deck.deal()
                current_hand.add_card(card)
                self.counter.update_count(card)

                card = self.deck.deal()
                new_hand.add_card(card)
                self.counter.update_count(card)

                self.player_hands.append(new_hand)
                self.bets.append(current_bet)
                # On a non-ace split, the player keeps playing the current
                # (now first) split hand. On an ace split, the player gets
                # no further actions -- step's "force STAND" guard will
                # advance through both split-aces hands automatically.
            else:
                # Can't split -- fall back to hit.
                card = self.deck.deal()
                current_hand.add_card(card)
                self.counter.update_count(card)
                if current_hand.is_busted():
                    advance_to_next = True

        elif action == 4:  # Late surrender
            # Allowed only on the first action of an un-split hand (2 cards,
            # current_hand_index == 0, only one hand in play). Otherwise
            # silently fall back to STAND (which is closest in spirit to
            # surrender being unavailable).
            if (len(current_hand.cards) == 2
                    and not current_hand.is_splitted
                    and self.current_hand_index == 0
                    and len(self.player_hands) == 1):
                current_hand.surrendered = True
            advance_to_next = True

        if advance_to_next:
            self.current_hand_index += 1

        self.counter.update_decks_remaining(self.deck.cards_remaining())

        # All player hands played? Then resolve the round.
        if self.current_hand_index >= len(self.player_hands):
            done = True
            self.game_over = True
            # Only deal the dealer's hole + draws if at least one player
            # hand is still live -- not busted AND not surrendered. If
            # everyone busted or surrendered, the dealer keeps the bets
            # without playing (and never reveals the hole).
            if any(not h.is_busted() and not h.surrendered for h in self.player_hands):
                self._deal_dealer()
            reward = self._calculate_rewards()
            self.bankroll += reward
            self.num_hands_played += 1
            return self._get_state(), reward, True

        return self._get_state(), 0, False
    
    def _deal_dealer(self):
        """
        Reveal the dealer's hole if not already revealed, then hit until
        the hand totals 17+ (H17: hits soft 17).
        """
        if not self.dealer_hole_dealt:
            # Standard path: deal the hole now.
            card = self.deck.deal()
            self.dealer_hand.add_card(card)
            self.counter.update_count(card)
        else:
            # The hole was already dealt during reset (US peek). The dealer
            # didn't have a natural (otherwise the round would already be
            # over), so the hole has been sitting face-down. Now reveal it.
            self.counter.update_count(self.dealer_hand.cards[1])

        # Hit until 17 or higher (H17: hit soft 17).
        while self.dealer_hand.value < 17 or (self.dealer_hand.value == 17 and self.dealer_hand.is_soft):
            card = self.deck.deal()
            self.dealer_hand.add_card(card)
            self.counter.update_count(card)

        self.counter.update_decks_remaining(self.deck.cards_remaining())
    
    def _calculate_rewards(self):
        """
        Calculate rewards for all player hands
        """
        total_reward = 0
        
        for i, hand in enumerate(self.player_hands):
            bet = self.bets[i]
            
            # Save the current_hand_index temporarily
            original_index = self.current_hand_index
            # Set to the current hand being processed
            self.current_hand_index = i
            
            if hand.surrendered:
                reward = -bet / 2.0  # forfeit half the bet
                result = "Surrender"
            elif hand.is_busted():
                reward = -bet
                result = "Bust"
            elif hand.is_blackjack and not self.dealer_hand.is_blackjack:
                reward = bet * 1.5  # 3:2 payout for blackjack
                result = "Blackjack"
            elif self.dealer_hand.is_busted():
                reward = bet
                result = "Dealer Bust"
            elif hand.is_blackjack and self.dealer_hand.is_blackjack:
                reward = 0  # Push
                result = "Push (Both Blackjack)"
            elif hand.value > self.dealer_hand.value:
                reward = bet
                result = "Win"
            elif hand.value < self.dealer_hand.value:
                reward = -bet
                result = "Lose"
            else:
                reward = 0  # Push
                result = "Push"
            
            total_reward += reward
            self._record_result(result, reward)
            
            # Restore the original index
            self.current_hand_index = original_index
        
        return total_reward
    
    def _record_result(self, result_type, reward):
        """
        Record game result for analysis
        """
        # Check if current_hand_index is valid before using it
        if self.current_hand_index < len(self.player_hands):
            player_hand = self.player_hands[self.current_hand_index]
        else:
            # Use the last hand if index is out of range
            player_hand = self.player_hands[-1] if self.player_hands else None
        
        self.results_history.append({
            'hand_num': self.num_hands_played,
            'result': result_type,
            'reward': reward,
            'running_count': self.counter.running_count,
            'true_count': self.counter.true_count,
            'bankroll': self.bankroll,
            'dealer_hand': str(self.dealer_hand),
            'player_hand': str(player_hand) if player_hand else "None"
        })


class BasicStrategyAgent:
    """
    Agent that uses basic strategy with card counting deviations
    """
    def __init__(self):
        # Basic strategy charts (simplified)
        # Format: player_value -> dealer_upcard -> action
        # Actions: 0 = hit, 1 = stand, 2 = double, 3 = split
        
        # Hard hands
        self.hard_strategy = {
            21: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            20: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            19: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            18: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            17: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            16: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            15: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            14: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            13: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            12: {2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            11: {2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2},
            10: {2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 0, 11: 0},
            9: {2: 0, 3: 2, 4: 2, 5: 2, 6: 2, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            8: {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            7: {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            6: {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            5: {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            4: {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
        }
        
        # Soft hands (hands with an Ace)
        self.soft_strategy = {
            21: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            20: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            19: {2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            18: {2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 1, 8: 1, 9: 0, 10: 0, 11: 0},
            17: {2: 0, 3: 2, 4: 2, 5: 2, 6: 2, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            16: {2: 0, 3: 0, 4: 2, 5: 2, 6: 2, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            15: {2: 0, 3: 0, 4: 2, 5: 2, 6: 2, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            14: {2: 0, 3: 0, 4: 0, 5: 2, 6: 2, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            13: {2: 0, 3: 0, 4: 0, 5: 2, 6: 2, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
        }
        
        # Pair splitting
        self.pair_strategy = {
            'A': {2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3},
            'K': {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            'Q': {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            'J': {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            '10': {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            '9': {2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 1, 8: 3, 9: 3, 10: 1, 11: 1},
            '8': {2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3},
            '7': {2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 0, 9: 0, 10: 0, 11: 0},
            '6': {2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            '5': {2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 0, 11: 0},
            '4': {2: 0, 3: 0, 4: 0, 5: 3, 6: 3, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            '3': {2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 0, 9: 0, 10: 0, 11: 0},
            '2': {2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 0, 9: 0, 10: 0, 11: 0},
        }
        
        # Deviations based on true count
        self.count_deviations = {
            # (player_value, dealer_upcard, default_action): (true_count_threshold, new_action)
            (16, 10, 0): (0, 1),  # Stand on 16 vs 10 if count >= 0
            (15, 10, 0): (4, 1),  # Stand on 15 vs 10 if count >= 4
            (10, 10, 0): (4, 2),  # Double on 10 vs 10 if count >= 4
            (12, 3, 0): (2, 1),   # Stand on 12 vs 3 if count >= 2
            (12, 2, 0): (3, 1),   # Stand on 12 vs 2 if count >= 3
            (11, 11, 2): (-1, 0), # Hit on 11 vs A if count < -1
            (9, 2, 0): (1, 2),    # Double on 9 vs 2 if count >= 1
            (10, 11, 0): (4, 2),  # Double on 10 vs A if count >= 4
            (9, 7, 0): (3, 2),    # Double on 9 vs 7 if count >= 3
            # Add more deviations as needed
        }
        
        # For tracking strategy deviations
        self.deviation_history = []
    
    def get_action(self, state):
        if not state:
            return 1  # Stand if no state (should not happen)
        
        player_hand = state['player_hand']
        player_value = state['player_value']
        dealer_up_card = state['dealer_up_card']
        is_soft = state['is_soft']
        can_split = state['can_split']
        true_count = state['true_count']
        
        # Record the default strategy and any deviations
        default_action = None
        deviation_made = False
        
        # Check if we can split
        if can_split and len(player_hand.cards) == 2:
            card_value = player_hand.cards[0].value
            default_action = self.pair_strategy.get(card_value, {}).get(dealer_up_card, 0)
            
            # Check for count-based deviations for pairs
            deviation_key = (f"pair_{card_value}", dealer_up_card, default_action)
            if deviation_key in self.count_deviations:
                threshold, new_action = self.count_deviations[deviation_key]
                if (threshold > 0 and true_count >= threshold) or (threshold <= 0 and true_count <= threshold):
                    deviation_made = True
                    action = new_action
                else:
                    action = default_action
            else:
                action = default_action
        
        # Check soft hands
        elif is_soft:
            default_action = self.soft_strategy.get(player_value, {}).get(dealer_up_card, 0)
            
            # Check for count-based deviations for soft hands
            deviation_key = (f"soft_{player_value}", dealer_up_card, default_action)
            if deviation_key in self.count_deviations:
                threshold, new_action = self.count_deviations[deviation_key]
                if (threshold > 0 and true_count >= threshold) or (threshold <= 0 and true_count <= threshold):
                    deviation_made = True
                    action = new_action
                else:
                    action = default_action
            else:
                action = default_action
        
        # Hard hands
        else:
            default_action = self.hard_strategy.get(player_value, {}).get(dealer_up_card, 0)
            
            # Check for count-based deviations for hard hands
            deviation_key = (player_value, dealer_up_card, default_action)
            if deviation_key in self.count_deviations:
                threshold, new_action = self.count_deviations[deviation_key]
                if (threshold > 0 and true_count >= threshold) or (threshold <= 0 and true_count <= threshold):
                    deviation_made = True
                    action = new_action
                else:
                    action = default_action
            else:
                action = default_action
        
        # Record deviation if it occurred
        if deviation_made:
            self.deviation_history.append({
                'player_value': player_value,
                'dealer_up_card': dealer_up_card,
                'is_soft': is_soft,
                'can_split': can_split,
                'true_count': true_count,
                'default_action': default_action,
                'chosen_action': action
            })
        
        return action