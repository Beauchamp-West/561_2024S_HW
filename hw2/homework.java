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
    public String calibratePath = "calibration.txt";

    private static final int[][] directions = new int[][] {{0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1}, {-1,0}, {-1,1}};
    private static final double[] searchTimes = new double[] {0.469, 0.129, 0.043, 0.027, 0.022, 0.019, 0.019};

    public Game(String inputPath) {
        restoreGameStateFromInput(inputPath);
        limit = calcLimit();
//        System.out.println(limit);
    }


    public void solve(String outputPath){
        int move = limit > 0 ? alphaBetaMove(limit) : randomMove();
        int row = move / 12, col = move % 12;
        String move_str = (char)('a'+col) + String.valueOf(1+row);
        output(move_str, outputPath);
    }

    public void solveTimer(int maxLevel) {
        int move = alphaBetaMove(maxLevel);
        int row = move / 12, col = move % 12;
        String move_str = (char)('a'+col) + String.valueOf(1+row);
        output(move_str, "timerTest.txt");
    }

    private int randomMove(){
        List<Integer> legalMoves = getLegalMoves(board, player);
        return legalMoves.get(new Random().nextInt(legalMoves.size()));
    }

    private int alphaBetaMove(int levelLimit){
        Map<Integer, List<Integer>> initMovesAndFlips = getLegalMovesAndFlips(board, player);
        Number[] valueAndMove = maxValue(board, player, initMovesAndFlips, -1d, 2d, 0, levelLimit);
        return (int) valueAndMove[1];
    }

    private Number[] maxValue(char[][] currBoard, char currPlayer, Map<Integer, List<Integer>> currMovesAndFlips, double alpha, double beta, int level, int limit) {
        if (currMovesAndFlips.isEmpty()) {
            // terminal state
            return getScore(currBoard) > 0 ? new Number[]{1d, -1} : new Number[]{0d, -1};
        } else if (level == limit) {
            // random evaluation
            return new Number[]{eval(currBoard, currPlayer, currMovesAndFlips), -1};
        }

        double v = -1d;
        int maxMove = -1;
        for (Map.Entry<Integer, List<Integer>> entry : currMovesAndFlips.entrySet()) {
            Integer move = entry.getKey();
            List<Integer> flipLocs = entry.getValue();
            char[][] nextBoard = getNextBoard(currBoard, currPlayer, flipLocs);
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
            return getScore(currBoard) > 0 ? new Number[]{1d, -1} : new Number[]{0d, -1};
        } else if (level == limit) {
            // random evaluation
            return new Number[]{eval(currBoard, currPlayer, currMovesAndFlips), -1};
        }

        double v = 2d;
        int minMove = -1;
        for (Map.Entry<Integer, List<Integer>> entry : currMovesAndFlips.entrySet()) {
            Integer move = entry.getKey();
            List<Integer> flipLocs = entry.getValue();
            char[][] nextBoard = getNextBoard(currBoard, currPlayer, flipLocs);
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

    private static boolean existLegalMove(char[][] board, char player) {
        boolean exist = false;
        for (int y = 0; y < 12; y++) {
            for (int x = 0; x < 12; x++) {
                if (board[y][x] == '.') {
                    for (int[] d : directions) {
                        int row = y + d[0], col = x + d[1];
                        boolean end = false;
                        boolean containsAnother = false;

                        while (row >= 0 && row < 12 && col >= 0 && col < 12) {
                            if (board[row][col] == '.') break;
                            else if (board[row][col] == player) {
                                end = true;
                                break;
                            } else {
                                containsAnother = true;
                            }
                            row += d[0];
                            col += d[1];
                        }

                        if (end && containsAnother) {
                            exist = true;
                            break;
                        }
                    }
                }
            }
        }

        return exist;
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

    private static char[][] getNextBoard(char[][] board, char player, List<Integer> flippedLocIndices) {
        char[][] newBoard = Arrays.stream(board).map(char[]::clone).toArray(char[][]::new);
        for (int idx : flippedLocIndices) {
            int row = idx / 12, col = idx % 12;
            newBoard[row][col] = player;
        }

        return newBoard;
    }

    private double eval(char[][] board, char player, Map<Integer, List<Integer>> movesAndFlips) {
        return new Random().nextDouble();
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

    private int calcLimit() {
        int maxSteps = countEmpty(board) / 2 + 1;
        double timePerStep = time_mine / maxSteps;
//        int limit = 0;
        int limit = searchTimes.length;
        for (double t : searchTimes) {
            if (t < timePerStep) break;
            limit--;
        }

//        try {
//            File inputFile = new File(calibratePath);
//            Scanner scanner = new Scanner(inputFile);
//
//            for (int i = maxLevel; i >= 1; i--) {
//                String time_str = scanner.nextLine();
//                double time = Double.parseDouble(time_str);
//                if (time < timePerStep) {
//                    limit = i;
//                    break;
//                }
//            }
//
//            scanner.close();
//        } catch (Exception e) {
//            e.printStackTrace();
//        }

        return limit;
    }

    private static int countEmpty(char[][] board) {
        int cnt = 0;
        for (int y = 0; y < 12; y++) {
            for (int x = 0; x < 12; x++) {
                if (board[y][x] == '.') cnt++;
            }
        }

        return cnt;
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
