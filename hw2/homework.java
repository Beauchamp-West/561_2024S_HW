import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class homework{
    public static void main(String[] args) {
        Game game = new Game("input.txt");
        game.solve("output.txt");
    }
}

class Game {
    private char[][] board;
    private char player;
    private int limit;
    private double time_mine, time_op;
    private double timePerNode = 1e-5;
    public String calibratePath = "calibration.txt";
    public int numExpandedNodes = 0;

    private static final int[][] directions = new int[][] {{0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1}, {-1,0}, {-1,1}};
//    private static final double[] searchTimes = new double[] {0.493, 0.272, 0.058, 0.037, 0.018, 0.018, 0.014, 0.021};
    private static final double[] searchTimes = new double[] {1d, 1d, 1d, 1d, 0.1, 0.1, 0.01, 0.01};

    public Game(String inputPath) {
        restoreGameStateFromInput(inputPath);
    }


    public void solve(String outputPath){
        int move = -1;
        Map<Integer, List<Integer>> initMovesAndFlips = getLegalMovesAndFlips(board, player);
        int branchingFactor = initMovesAndFlips.size();
        if (branchingFactor == 1) {
            for (Integer i : initMovesAndFlips.keySet()) {
                move = i;
                break;
            }
        } else {
            if (branchingFactor < 4) limit = 1;
            else if (branchingFactor < 10) limit = 2;
            else limit = 3;
            move = alphaBetaMove(limit, initMovesAndFlips);
        }

        System.out.println(branchingFactor);
        System.out.println(limit);
        int row = move / 12, col = move % 12;
        String move_str = (char)('a'+col) + String.valueOf(1+row);
        output(move_str, outputPath);
    }

    public void solveTimer(int maxLevel) {
        Map<Integer, List<Integer>> initMovesAndFlips = getLegalMovesAndFlips(board, player);
        int move = alphaBetaMove(maxLevel, initMovesAndFlips);
        int row = move / 12, col = move % 12;
        String move_str = (char)('a'+col) + String.valueOf(1+row);
        output(move_str, "timerTest.txt");
    }

    private int alphaBetaMove(int levelLimit, Map<Integer, List<Integer>> initMovesAndFlips){
        Number[] valueAndMove = maxValue(board, player, initMovesAndFlips, -101d, 101d, 0, levelLimit);
        return (int) valueAndMove[1];
    }

    private Number[] maxValue(char[][] currBoard, char currPlayer, Map<Integer, List<Integer>> currMovesAndFlips, double alpha, double beta, int level, int limit) {
        if (currMovesAndFlips.isEmpty()) {
            // terminal state
            return getScore(currBoard) > 0 ? new Number[]{100d, -1} : new Number[]{-100d, -1};
        } else if (level == limit) {
            // random evaluation
            return new Number[]{eval(currBoard, currPlayer, currMovesAndFlips), -1};
        }

        double v = -101d;
        int maxMove = -1;
        for (Map.Entry<Integer, List<Integer>> entry : currMovesAndFlips.entrySet()) {
            numExpandedNodes++;
            Integer move = entry.getKey();
            List<Integer> flipLocs = entry.getValue();
            char[][] nextBoard = getNextBoard(currBoard, currPlayer, flipLocs, move);
            char nextPlayer = currPlayer == 'X' ? 'O' : 'X';
            Map<Integer, List<Integer>> nextMovesAndFlips = getLegalMovesAndFlips(nextBoard, nextPlayer);

            double nextV = nextMovesAndFlips.isEmpty() ?
                    (double) maxValue(nextBoard, currPlayer, getLegalMovesAndFlips(nextBoard, currPlayer), alpha, beta, level+1, limit)[0] :
                    (double) minValue(nextBoard, nextPlayer, nextMovesAndFlips, alpha, beta, level+1, limit)[0];
            if (nextV > v) {
                v = nextV;
                maxMove = move;
            }
            if (v >= beta) return new Number[] {v, -1};
            alpha = Math.max(alpha, v);
        }

        return new Number[]{v, maxMove};
    }

    private Number[] minValue(char[][] currBoard, char currPlayer, Map<Integer, List<Integer>> currMovesAndFlips, double alpha, double beta, int level, int limit) {
        if (currMovesAndFlips.isEmpty()) {
            // terminal state
            return getScore(currBoard) > 0 ? new Number[]{100d, -1} : new Number[]{-100d, -1};
        } else if (level == limit) {
            // random evaluation
            return new Number[]{eval(currBoard, currPlayer, currMovesAndFlips), -1};
        }

        double v = 101d;
        int minMove = -1;
        for (Map.Entry<Integer, List<Integer>> entry : currMovesAndFlips.entrySet()) {
            numExpandedNodes++;
            Integer move = entry.getKey();
            List<Integer> flipLocs = entry.getValue();
            char[][] nextBoard = getNextBoard(currBoard, currPlayer, flipLocs, move);
            char nextPlayer = currPlayer == 'X' ? 'O' : 'X';
            Map<Integer, List<Integer>> nextMovesAndFlips = getLegalMovesAndFlips(nextBoard, nextPlayer);

            double nextV = nextMovesAndFlips.isEmpty() ?
                    (double) minValue(nextBoard, currPlayer, getLegalMovesAndFlips(nextBoard, currPlayer), alpha, beta, level+1, limit)[0] :
                    (double) maxValue(nextBoard, nextPlayer, nextMovesAndFlips, alpha, beta, level+1, limit)[0];
            if (nextV < v) {
                v = nextV;
                minMove = move;
            }
            if (v <= alpha) return new Number[] {v, -1};
            beta = Math.min(beta, v);
        }

        return new Number[]{v, minMove};
    }

    private static List<Integer> getLegalMoves(char[][] board, char player) {
        List<Integer> moves = new ArrayList<>();
        for (int y = 0; y < 12; y++) {
            for (int x = 0; x < 12; x++) {
                if (board[y][x] == '.') {
                    for (int[] d : directions) {
                        int row = y + d[0], col = x + d[1];
                        boolean end = false;
                        List<Integer> tmpFlipCandidates = new ArrayList<>();

                        while (row >= 0 && row < 12 && col >= 0 && col < 12) {
                            if (board[row][col] == '.') break;
                            else if (board[row][col] == player) {
                                end = true;
                                break;
                            } else {
                                Integer idx = row * 12 + col;
                                tmpFlipCandidates.add(idx);
                            }
                            row += d[0];
                            col += d[1];
                        }

                        if (end && !tmpFlipCandidates.isEmpty()) {
                            moves.add(y*12+x);
                            break;
                        }
                    }
                }
            }
        }

        return moves;
    }

    private static Map<Integer, List<Integer>> getLegalMovesAndFlips(char[][] board, char player) {
        Map<Integer, List<Integer>> move2Flips = new HashMap<>();
        for (int y = 0; y < 12; y++) {
            for (int x = 0; x < 12; x++) {
                if (board[y][x] == '.') {
                    List<Integer> flipCandidates = new ArrayList<>();
                    for (int[] d : directions) {
                        int row = y + d[0], col = x + d[1];
                        boolean end = false;
                        List<Integer> tmpFlipCandidates = new ArrayList<>();

                        while (row >= 0 && row < 12 && col >= 0 && col < 12) {
                            if (board[row][col] == '.') break;
                            else if (board[row][col] == player) {
                                end = true;
                                break;
                            } else {
                                Integer idx = row * 12 + col;
                                tmpFlipCandidates.add(idx);
                            }
                            row += d[0];
                            col += d[1];
                        }

                        if (end) flipCandidates.addAll(tmpFlipCandidates);
                    }

                    if (!flipCandidates.isEmpty()) {
                        Integer idx = y * 12 + x;
                        move2Flips.put(idx, flipCandidates);
                    }
                }
            }
        }

        return move2Flips;
    }

    private static char[][] getNextBoard(char[][] board, char player, List<Integer> flippedLocIndices, Integer move) {
        char[][] newBoard = Arrays.stream(board).map(char[]::clone).toArray(char[][]::new);
        for (int idx : flippedLocIndices) {
            int row = idx / 12, col = idx % 12;
            newBoard[row][col] = player;
        }
        int r = move / 12, c = move % 12;
        newBoard[r][c] = player;

        return newBoard;
    }

    private double eval(char[][] board, char player, Map<Integer, List<Integer>> movesAndFlips) {
//        return new Random().nextDouble() * 200d - 100d;
        int[] moveCnt = countMoves(board, player, movesAndFlips);
        double parityHeuristic = 100 * (moveCnt[0] - moveCnt[1]) / (double) (moveCnt[0] + moveCnt[1]);
        double moveHeuristic = 100 * (moveCnt[2] - moveCnt[3]) / (double) (moveCnt[2] + moveCnt[3]);
        int totalPotentialMoves = moveCnt[4] + moveCnt[5];
        double potentialMoveHeuristic = totalPotentialMoves > 0 ? 100 * (moveCnt[4] - moveCnt[5]) / (double) totalPotentialMoves : 0d;

        int[] cornerCnt = countCorners(board);
        int totalCorners = cornerCnt[0] + cornerCnt[1], totalCornerNeighbors = cornerCnt[2] + cornerCnt[3];
        double cornerHeuristic = totalCorners > 0 ? 100 * (cornerCnt[0] - cornerCnt[1]) / (double) totalCorners : 0d;
        double closeCornerHeuristic = totalCornerNeighbors > 0 ? -100 * (cornerCnt[2] - cornerCnt[3]) / (double) totalCornerNeighbors : 0d;

        return 0.01 * parityHeuristic + 0.07 * (moveHeuristic + potentialMoveHeuristic) + 0.15 * closeCornerHeuristic + 0.7 * cornerHeuristic;
    }

    private int[] countMoves(char[][] board, char currPlayer, Map<Integer, List<Integer>> currMovesAndFlips) {
        int numPlayers = 0, numOps = 0, numPlayerPotentialMoves = 0, numOpPotentialMoves = 0;
        char op = player == 'X' ? 'O' : 'X';
        boolean isCurrPlayer = currPlayer == player;
        char another = isCurrPlayer ? op : player;
        int numCurrMoves = currMovesAndFlips.size(), numAnotherMoves = 0;
        for (int y = 0; y < 12; y++) {
            for (int x = 0; x < 12; x++) {
                if (board[y][x] == '.') {
                    // check another one's move and potential move
                    boolean isPlayerPotentialMove = false, isOpPotentialMove = false;
                    for (int[] d : directions) {
                        int row = y + d[0], col = x + d[1];
                        if (row >= 0 && row < 12 && col >= 0 && col < 12) {
                            if (board[row][col] == player) isOpPotentialMove = true;
                            else if (board[row][col] == op) isPlayerPotentialMove = true;
                        }
                        boolean anotherEnd = false, containsCurr = false;
                        while (row >= 0 && row < 12 && col >= 0 && col < 12) {
                            if (board[row][col] == '.') break;
                            else if (board[row][col] == another) {
                                anotherEnd = true;
                                break;
                            } else {
                                containsCurr = true;
                            }
                            row += d[0];
                            col += d[1];
                        }
                        if (anotherEnd && containsCurr) {
                            numAnotherMoves++;
                            break;
                        }
                    }
                    if (isPlayerPotentialMove) numPlayerPotentialMoves++;
                    if (isOpPotentialMove) numOpPotentialMoves++;
                } else if (board[y][x] == player) {
                    // update player number
                    numPlayers++;
                } else {
                    // update opponent number
                    numOps++;
                }
            }
        }

        if (isCurrPlayer) {
            return new int[] {numPlayers, numOps, numCurrMoves, numAnotherMoves, numPlayerPotentialMoves, numOpPotentialMoves};
        } else {
            return new int[] {numPlayers, numOps, numAnotherMoves, numCurrMoves, numPlayerPotentialMoves, numOpPotentialMoves};
        }
    }

    private int[] countCorners(char[][] board) {
        int numPlayersCorner = 0, numOpsCorner = 0, numPlayersCloseCorner = 0, numOpsCloseCorner = 0;
        char op = player == 'X' ? 'O' : 'X';
        int[][] corners = new int[][] {{0,0}, {11,0}, {0,11}, {11,11}};
        int[][] cornerNeighbors = new int[][] {{0,1}, {1,0}, {1,1}, {11,1}, {10,0}, {10,1}, {0,10}, {1,11}, {1,10}, {11,10}, {10,11}, {10,10}};
        for (int[] loc : corners) {
            if (board[loc[0]][loc[1]] == player) numPlayersCorner++;
            else if (board[loc[0]][loc[1]] == op) numOpsCorner++;
        }
        for (int[] loc : cornerNeighbors) {
            if (board[loc[0]][loc[1]] == player) numPlayersCloseCorner++;
            else if (board[loc[0]][loc[1]] == op) numOpsCloseCorner++;
        }

        return new int[] {numPlayersCorner, numOpsCorner, numPlayersCloseCorner, numOpsCloseCorner};
    }


    private int getScore(char[][] board) {
        int numPlayer = player == 'X' ? 1 : 0;
        int numOp = 1 - numPlayer;
        for (int y = 0; y < 12; y++) {
            for (int x = 0; x < 12; x++) {
                if (board[y][x] == player) numPlayer++;
                else if (board[y][x] != '.') numOp++;
            }
        }

        return numPlayer - numOp;
    }

    public static void output(String s, String outputPath) {
        try {
            FileWriter writer = new FileWriter(outputPath);
            writer.write(s);
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void restoreGameStateFromInput(String inputPath) {
        try {
            File inputFile = new File(inputPath);
            Scanner scanner = new Scanner(inputFile);

            String player_str = scanner.nextLine();
            player = player_str.toCharArray()[0];
            String[] times = scanner.nextLine().split(" ");
            time_mine = Double.parseDouble(times[0]);
            time_op = Double.parseDouble(times[1]);

            board = new char[12][12];
            for (int i = 0; i < 12; i++) {
                board[i] = scanner.nextLine().toCharArray();
            }
            scanner.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
