package engine.core;

import java.io.*;
import java.awt.image.VolatileImage;
import java.util.ArrayList;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.io.FileWriter; 
import java.io.File;

import javax.swing.JFrame;

import agents.human.Agent;
import engine.helper.GameStatus;
import engine.helper.MarioActions;

import engine.core.StepResult;
import java.awt.event.WindowEvent;

public class MarioGame {
    /**
     * the maximum time that agent takes for each step
     */
    public static final long maxTime = 40;
    /**
     * extra time before reporting that the agent is taking more time that it should
     */
    public static final long graceTime = 10;
    /**
     * Screen width
     */
    public static final int width = 256;
    /**
     * Screen height
     */
    public static final int height = 256;
    /**
     * Screen width in tiles
     */
    public static final int tileWidth = width / 16;
    /**
     * Screen height in tiles
     */
    public static final int tileHeight = height / 16;
    /**
     * print debug details
     */
    public static final boolean verbose = false;

    /**
     * pauses the whole game at any moment
     */
    public boolean pause = false;

    /**
     * events that kills the player when it happens only care about type and param
     */
    private MarioEvent[] killEvents;

    //visualization
    private JFrame window = null;
    private MarioRender render = null;
    private MarioAgent agent = null;
    private MarioWorld world = null;

 // For agent control in steps
    private MarioTimer agentTimer;
    private boolean visuals;
    private VolatileImage renderTarget;
    private Graphics backBuffer;
    private Graphics currentBuffer;
    private ArrayList<MarioEvent> gameEvents;
    private ArrayList<MarioAgentEvent> agentEvents;
    private int fps;
    private long currentTime;
    
    /**
     * Create a mario game to be played
     */
    public MarioGame() {

    }

    /**
     * Create a mario game with a different forward model where the player on certain event
     *
     * @param killEvents events that will kill the player
     */
    public MarioGame(MarioEvent[] killEvents) {
        this.killEvents = killEvents;
    }

    private int getDelay(int fps) {
        if (fps <= 0) {
            return 0;
        }
        return 1000 / fps;
    }

    private void setAgent(MarioAgent agent) {
        this.agent = agent;
        if (agent instanceof KeyAdapter) {
            this.render.addKeyListener((KeyAdapter) this.agent);
        }
    }

    /**
     * Play a certain mario level
     *
     * @param level a string that constitutes the mario level, it uses the same representation as the VGLC but with more details. for more details about each symbol check the json file in the levels folder.
     * @param timer number of ticks for that level to be played. Setting timer to anything &lt;=0 will make the time infinite
     * @return statistics about the current game
     */
    public MarioResult playGame(String level, int timer) {
        return this.runGame(new Agent(), level, timer, 0, true, 30, 2);
    }

    /**
     * Play a certain mario level
     *
     * @param level      a string that constitutes the mario level, it uses the same representation as the VGLC but with more details. for more details about each symbol check the json file in the levels folder.
     * @param timer      number of ticks for that level to be played. Setting timer to anything &lt;=0 will make the time infinite
     * @param marioState the initial state that mario appears in. 0 small mario, 1 large mario, and 2 fire mario.
     * @return statistics about the current game
     */
    public MarioResult playGame(String level, int timer, int marioState) {
        return this.runGame(new Agent(), level, timer, marioState, true, 30, 2);
    }

    /**
     * Play a certain mario level
     *
     * @param level      a string that constitutes the mario level, it uses the same representation as the VGLC but with more details. for more details about each symbol check the json file in the levels folder.
     * @param timer      number of ticks for that level to be played. Setting timer to anything &lt;=0 will make the time infinite
     * @param marioState the initial state that mario appears in. 0 small mario, 1 large mario, and 2 fire mario.
     * @param fps        the number of frames per second that the update function is following
     * @return statistics about the current game
     */
    public MarioResult playGame(String level, int timer, int marioState, int fps) {
        return this.runGame(new Agent(), level, timer, marioState, true, fps, 2);
    }

    /**
     * Play a certain mario level
     *
     * @param level      a string that constitutes the mario level, it uses the same representation as the VGLC but with more details. for more details about each symbol check the json file in the levels folder.
     * @param timer      number of ticks for that level to be played. Setting timer to anything &lt;=0 will make the time infinite
     * @param marioState the initial state that mario appears in. 0 small mario, 1 large mario, and 2 fire mario.
     * @param fps        the number of frames per second that the update function is following
     * @param scale      the screen scale, that scale value is multiplied by the actual width and height
     * @return statistics about the current game
     */
    public MarioResult playGame(String level, int timer, int marioState, int fps, float scale) {
        return this.runGame(new Agent(), level, timer, marioState, true, fps, scale);
    }

    /**
     * Run a certain mario level with a certain agent
     *
     * @param agent the current AI agent used to play the game
     * @param level a string that constitutes the mario level, it uses the same representation as the VGLC but with more details. for more details about each symbol check the json file in the levels folder.
     * @param timer number of ticks for that level to be played. Setting timer to anything &lt;=0 will make the time infinite
     * @return statistics about the current game
     */
    public MarioResult runGame(MarioAgent agent, String level, int timer) {
        return this.runGame(agent, level, timer, 0, false, 0, 2);
    }

    /**
     * Run a certain mario level with a certain agent
     *
     * @param agent      the current AI agent used to play the game
     * @param level      a string that constitutes the mario level, it uses the same representation as the VGLC but with more details. for more details about each symbol check the json file in the levels folder.
     * @param timer      number of ticks for that level to be played. Setting timer to anything &lt;=0 will make the time infinite
     * @param marioState the initial state that mario appears in. 0 small mario, 1 large mario, and 2 fire mario.
     * @return statistics about the current game
     */
    public MarioResult runGame(MarioAgent agent, String level, int timer, int marioState) {
        return this.runGame(agent, level, timer, marioState, false, 0, 2);
    }

    /**
     * Run a certain mario level with a certain agent
     *
     * @param agent      the current AI agent used to play the game
     * @param level      a string that constitutes the mario level, it uses the same representation as the VGLC but with more details. for more details about each symbol check the json file in the levels folder.
     * @param timer      number of ticks for that level to be played. Setting timer to anything &lt;=0 will make the time infinite
     * @param marioState the initial state that mario appears in. 0 small mario, 1 large mario, and 2 fire mario.
     * @param visuals    show the game visuals if it is true and false otherwise
     * @return statistics about the current game
     */
    public MarioResult runGame(MarioAgent agent, String level, int timer, int marioState, boolean visuals) {
        return this.runGame(agent, level, timer, marioState, visuals, visuals ? 30 : 0, 2);
    }

    /**
     * Run a certain mario level with a certain agent
     *
     * @param agent      the current AI agent used to play the game
     * @param level      a string that constitutes the mario level, it uses the same representation as the VGLC but with more details. for more details about each symbol check the json file in the levels folder.
     * @param timer      number of ticks for that level to be played. Setting timer to anything &lt;=0 will make the time infinite
     * @param marioState the initial state that mario appears in. 0 small mario, 1 large mario, and 2 fire mario.
     * @param visuals    show the game visuals if it is true and false otherwise
     * @param fps        the number of frames per second that the update function is following
     * @return statistics about the current game
     */
    public MarioResult runGame(MarioAgent agent, String level, int timer, int marioState, boolean visuals, int fps) {
        return this.runGame(agent, level, timer, marioState, visuals, fps, 2);
    }

    /**
     * Custom function to create a new mario game instance 
     * 
     * 
     * 
    */
    public MarioResult buildGame(MarioAgent agent, String level, int timer, int marioState, boolean visuals) {
        return this.resetGame(agent, level, timer, marioState, visuals, visuals ? 30 : 0, 2);
    }

    /**
     * Initializes a new game level
     * @param agent
     * @param level
     * @param timer
     * @param marioState
     * @param visuals
     * @param fps
     * @param scale
     * @return
     */
    public MarioResult resetGame(MarioAgent agent, String level, int timer, int marioState, boolean visuals, int fps, float scale) {
        this.visuals = visuals;
        this.fps = fps;
        if (visuals) {
            this.window = new JFrame("Mario AI Framework - python");
            this.render = new MarioRender(scale);
            this.window.setContentPane(this.render);
            this.window.pack();
            this.window.setResizable(false);
            this.window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            this.render.init();
            this.window.setVisible(true);
        }
        this.setAgent(agent);
        // return this.gameLoop(level, timer, marioState, visuals, fps);
        this.world = new MarioWorld(this.killEvents);
        this.world.visuals = visuals;
        this.world.initializeLevel(level, 1000 * timer);
        if (visuals) {
            this.world.initializeVisuals(this.render.getGraphicsConfiguration());
        }
        this.world.mario.isLarge = marioState > 0;
        this.world.mario.isFire = marioState > 1;
        this.world.update(new boolean[MarioActions.numberOfActions()]);
        this.currentTime = System.currentTimeMillis();

        //initialize graphics
    
        if (visuals) {
            this.renderTarget = this.render.createVolatileImage(MarioGame.width, MarioGame.height);
            this.backBuffer = this.render.getGraphics();
            this.currentBuffer = renderTarget.getGraphics();
            this.render.addFocusListener(this.render);
        }

        this.agentTimer = new MarioTimer(MarioGame.maxTime);
        this.agent.initialize(new MarioForwardModel(this.world.clone()), this.agentTimer);

        this.gameEvents = new ArrayList<>();
        this.agentEvents = new ArrayList<>();


        return new MarioResult(this.world, this.gameEvents, this.agentEvents);


    }

    /** Step function */

    public StepResult stepGame() {
        boolean[] actions = this.agent.getActions(new MarioForwardModel(this.world.clone()), agentTimer); // replace with agent action
        return stepGame(actions);
    }

    public StepResult stepGame(boolean[] actions) {

        MarioForwardModel model = new MarioForwardModel(this.world);
        // while (this.world.gameStatus == GameStatus.RUNNING) {
            if (!this.pause) {
                //get actions
                agentTimer = new MarioTimer(MarioGame.maxTime);
                if (MarioGame.verbose) {
                    if (agentTimer.getRemainingTime() < 0 && Math.abs(agentTimer.getRemainingTime()) > MarioGame.graceTime) {
                        System.out.println("The Agent is slowing down the game by: "
                                + Math.abs(this.agentTimer.getRemainingTime()) + " msec.");
                    }
                }
                // update world
                this.world.update(actions);
                this.gameEvents.addAll(this.world.lastFrameEvents);
                this.agentEvents.add(new MarioAgentEvent(actions, this.world.mario.x,
                        this.world.mario.y, (this.world.mario.isLarge ? 1 : 0) + (this.world.mario.isFire ? 1 : 0),
                        this.world.mario.onGround, this.world.currentTick));
            }
            // Gym setup based on https://github.com/Kautenja/gym-super-mario-bros 
            // get obs
            int[][] obs = model.getScreenCompleteObservation(0,0);

            int[] pos = model.getMarioScreenTilePos();
//            System.out.println("Coords: " + pos[0] + ", " + pos[1]);
            obs[Math.min(pos[0], 15)][Math.min(pos[1],15)] = 99;

            // print obs

//            for (int[] x : obs) {
//                for (int y : x) {
//                        System.out.print(y + " ");
//                }
//                System.out.println();
//            }
//            System.out.println("----------------");
            
            // get reward
            float vx = model.getMarioFloatVelocity()[0];
            float vx2 = world.mario.xa;
            float c = this.getDelay(this.fps);
            float c2 = world.currentTick;
            int d = world.mario.alive ? 0 : -15;

            boolean done = world.gameStatus ==  GameStatus.WIN || world.gameStatus == GameStatus.LOSE;
            float f = model.getCompletionPercentage() == 1? 15 : 0;
            
//            System.out.println("vx: " + vx + ", " + vx2 +
//            "\n c: " + c + ", " + c2 +
//            "\n d: " + d
//            );

            float reward = vx2 + d + f - 1;

            reward = reward > 15 ? 15 : (reward < -15 ? -15 : reward);
//            System.out.println("reward: " + reward);
//
//            System.out.println("done: " + done);


            //render world
            if (this.visuals) {
                this.render.renderWorld(this.world, this.renderTarget, this.backBuffer, this.currentBuffer);
            }

            

            //check if delay needed
            if (this.getDelay(this.fps) > 0) {
                try {
                    this.currentTime += this.getDelay(this.fps);
                    Thread.sleep(Math.max(0, this.currentTime - System.currentTimeMillis()));
                } catch (InterruptedException e) {
                    // break;
                }
            }
            


        return new StepResult(obs, reward, done, "");
        // }
        // return new MarioResult(this.world, gameEvents, agentEvents);
    }

    public void killGame() {
        // this.window.dispatchEvent(new WindowEvent(window, WindowEvent.WINDOW_CLOSING));
        this.window.dispose();
    }


    
    /**
     * Run a certain mario level with a certain agent
     *
     * @param agent      the current AI agent used to play the game
     * @param level      a string that constitutes the mario level, it uses the same representation as the VGLC but with more details. for more details about each symbol check the json file in the levels folder.
     * @param timer      number of ticks for that level to be played. Setting timer to anything &lt;=0 will make the time infinite
     * @param marioState the initial state that mario appears in. 0 small mario, 1 large mario, and 2 fire mario.
     * @param visuals    show the game visuals if it is true and false otherwise
     * @param fps        the number of frames per second that the update function is following
     * @param scale      the screen scale, that scale value is multiplied by the actual width and height
     * @return statistics about the current game
     */
    public MarioResult runGame(MarioAgent agent, String level, int timer, int marioState, boolean visuals, int fps, float scale) {
        if (visuals) {
            this.window = new JFrame("Mario AI Framework");
            this.render = new MarioRender(scale);
            this.window.setContentPane(this.render);
            this.window.pack();
            this.window.setResizable(false);
            this.window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            this.render.init();
            this.window.setVisible(true);
        }
        this.setAgent(agent);
        return this.gameLoop(level, timer, marioState, visuals, fps);
    }

    private MarioResult gameLoop(String level, int timer, int marioState, boolean visual, int fps) {
        this.world = new MarioWorld(this.killEvents);
        this.world.visuals = visual;
        this.world.initializeLevel(level, 1000 * timer);
        if (visual) {
            this.world.initializeVisuals(this.render.getGraphicsConfiguration());
        }
        this.world.mario.isLarge = marioState > 0;
        this.world.mario.isFire = marioState > 1;
        this.world.update(new boolean[MarioActions.numberOfActions()]);
        long currentTime = System.currentTimeMillis();

        //initialize graphics
        VolatileImage renderTarget = null;
        Graphics backBuffer = null;
        Graphics currentBuffer = null;
        if (visual) {
            renderTarget = this.render.createVolatileImage(MarioGame.width, MarioGame.height);
            backBuffer = this.render.getGraphics();
            currentBuffer = renderTarget.getGraphics();
            this.render.addFocusListener(this.render);
        }

        MarioTimer agentTimer = new MarioTimer(MarioGame.maxTime);

        MarioForwardModel model =  new MarioForwardModel(this.world.clone());
        this.agent.initialize(model, agentTimer);

        ArrayList<MarioEvent> gameEvents = new ArrayList<>();
        ArrayList<MarioAgentEvent> agentEvents = new ArrayList<>();

        File PositionData = new File("positionData.txt"); 
        PositionData.delete(); 
        File PositionData2 = new File("positionData.txt"); 

        // Print x-length of the level 

        int xLength = level.length()/16-1; 
        String xLengthString = Integer.toString(xLength) + "\n";
        try{
            FileWriter myWriter = new FileWriter("positionData.txt", true);
            myWriter.write(xLengthString);
            myWriter.close();
        }
        catch(IOException ex){
            System.out.println("An error occured");
            return new MarioResult(this.world, gameEvents, agentEvents);
        }    
        while (this.world.gameStatus == GameStatus.RUNNING) {
            if (!this.pause) {
                //get actions
                agentTimer = new MarioTimer(MarioGame.maxTime);
                boolean[] actions = this.agent.getActions(model, agentTimer);
                if (MarioGame.verbose) {
                    if (agentTimer.getRemainingTime() < 0 && Math.abs(agentTimer.getRemainingTime()) > MarioGame.graceTime) {
                        System.out.println("The Agent is slowing down the game by: "
                                + Math.abs(agentTimer.getRemainingTime()) + " msec.");
                    }
                }
                // update world
                this.world.update(actions);
                gameEvents.addAll(this.world.lastFrameEvents);
                agentEvents.add(new MarioAgentEvent(actions, this.world.mario.x,
                        this.world.mario.y, (this.world.mario.isLarge ? 1 : 0) + (this.world.mario.isFire ? 1 : 0),
                        this.world.mario.onGround, this.world.currentTick));
            }

            //System.out.println("Current state of agent " + model.getMarioScreenTilePos()[0] + ", " +model.getMarioScreenTilePos()[1]);
            System.out.println("Current Mario Position " + this.world.mario.x +", " + this.world.mario.y);

            int screenX = (int) (this.world.mario.x  / 16);
            int screenY = (int) (this.world.mario.y / 16);
            String marioX = Integer.toString(screenX);
            String marioY = Integer.toString(screenY); 
            String marioCoordinates = marioX + "," + marioY+"\n"; 
            System.out.println(marioCoordinates);
            try{
                FileWriter myWriter = new FileWriter("positionData.txt", true);
                myWriter.write(marioCoordinates);
                myWriter.close();
            }
            catch(IOException ex){
                System.out.println("An error occured");
                return new MarioResult(this.world, gameEvents, agentEvents);
            }            

            //render world
            if (visual) {
                this.render.renderWorld(this.world, renderTarget, backBuffer, currentBuffer);
            }
            //check if delay needed
            if (this.getDelay(fps) > 0) {
                try {
                    currentTime += this.getDelay(fps);
                    Thread.sleep(Math.max(0, currentTime - System.currentTimeMillis()));
                } catch (InterruptedException e) {
                    break;
                }
            }
        }
        return new MarioResult(this.world, gameEvents, agentEvents);
    }
}
