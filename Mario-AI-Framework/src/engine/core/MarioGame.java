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
import engine.helper.EventType;
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
        
        // Set up trajectory collection for imitation learning models
        ArrayList<int[][]> obsList = new ArrayList<>();
        ArrayList<boolean[]> actList = new ArrayList<>(); 
        ArrayList<float[]> infoList = new ArrayList<>();
        
        // Collect the first obs because length of obsList must be len(actList + 1)
        int[][] obs = model.getScreenCompleteObservation(0,0); // NOTE: Went back to using fwd model instead of world
        int[] pos = model.getMarioScreenTilePos();
        obs[Math.min(15,pos[0])][Math.min(15, pos[1])] = 98 + this.world.mario.facing; // Place mario tile and keep track of where he is facing
        obsList.add(obs);
        
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
            //System.out.println("Current Mario Position " + this.world.mario.x +", " + this.world.mario.y);

            int screenX = (int) (this.world.mario.x  / 16);
            int screenY = (int) (this.world.mario.y / 16);
            String marioX = Integer.toString(screenX);
            String marioY = Integer.toString(screenY); 
            String marioCoordinates = marioX + "," + marioY+"\n"; 
            //System.out.println(marioCoordinates);
            
            /*---------Trajectory Collection----------*/
            // Collect obs, rew, and info for trajectory
            
            // Obs
            obs = model.getScreenCompleteObservation(0,0); // NOTE: Went back to using fwd model instead of world
            pos = model.getMarioScreenTilePos();
            obs[Math.min(15,pos[0])][Math.min(15, pos[1])] = 98 + this.world.mario.facing; // Place mario tile and keep track of where he is facing
            
            // Info
            int g = 0; // Count of how many enemies were stomped
            for (MarioEvent e : gameEvents) {
                if (e.getEventType() == EventType.STOMP_KILL.getValue()) {
                    g += 1;
                }
            }
            
        	float x = world.mario.x; // x position
            float vx = world.mario.xa; // instantaneous velocity 
            int d = world.mario.alive ? 0 : -15; // penalty for dying           
            float f = model.getCompletionPercentage() == 1? 15 : 0; // reward for full completion
            int c = 4; // assume difference in clock is 4 frames due to 4-frame skip
            
            float[] info = new float[] {vx, d, f, c, x, g};
            infoList.add(info);
            
            // Agent is rewarded for moving to the right, while not dying, and doing so as quick as possible
            float reward = vx + d + f - c;  // NOTE: DONT NEED THIS
         
            /*----------------------------------------*/
           
            
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
        return new MarioResult(this.world, gameEvents, agentEvents, obsList, actList, infoList);
    }

 // ************************ CUSTOM FUNCTIONS - Michael *******************************

    /**
     * Create a new instance of a Mario game (based on runGame functions from original)
    */
    public MarioResult buildGame(MarioAgent agent, String level, int timer, int marioState, boolean visuals) {
        return this.resetGame(agent, level, timer, marioState, visuals, visuals ? 30 : 0, 2);
    }

    /**
     * Initializes a new game level (based on gameLoop function from original)
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
        // Modified original framework code to convert some variables to instance variables so they can be accessed elsewhere
    	
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
        
        // Print first line of positionData.txt
        
        File PositionData = new File("src/positionData.txt"); 
        PositionData.delete(); 
        File PositionData2 = new File("src/positionData.txt"); 

        // Print x-length of the level 

        int xLength = level.length()/16-1; 
        String xLengthString = Integer.toString(xLength) + "\n";
        try{
            FileWriter myWriter = new FileWriter("src/positionData.txt", true);
            myWriter.write(xLengthString);
            myWriter.close();
        }
        catch(IOException ex){
            System.out.println("An error occured");
            return new MarioResult(this.world, gameEvents, agentEvents);
        }    

        return new MarioResult(this.world, this.gameEvents, this.agentEvents);
    }

    /** Step function */
    // This is the core of the environment code
    
    public StepResult stepGame() {
        // Run this if we're using an A* or User as input action

        boolean[] actions = this.agent.getActions(new MarioForwardModel(this.world.clone()), agentTimer); // replace with agent action
        return stepGame(actions);
    }

    public StepResult stepGame(boolean[] actions) {
        // Run this to step game from python code

        MarioForwardModel model = new MarioForwardModel(this.world);

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

        	int screenX = (int) (this.world.mario.x  / 16);
            int screenY = (int) (this.world.mario.y / 16);
            String marioX = Integer.toString(screenX);
            String marioY = Integer.toString(screenY); 
            String marioCoordinates = marioX + "," + marioY+"\n"; 
        	
            // Gym setup based on https://github.com/Kautenja/gym-super-mario-bros 
//            // get obs - seems like 
            int[][] obs = model.getScreenCompleteObservation(0,0);
            int[] pos = model.getMarioScreenTilePos();
            
//            // TODO: Switch this with world model for [-1] offset!
//            int[][] obs = this.world.getMergedObservation(this.world.cameraX  + MarioGame.width / 2, MarioGame.height / 2,0,0);
//            int[] pos = new int[] {(int)((this.world.mario.x - this.world.cameraX) / 16), (int) (this.world.mario.y / 16)};
//            int[] pos = model.getMarioScreenTilePos();
            
            
            obs[Math.min(15,pos[0])][Math.min(15, pos[1])] = 98 + this.world.mario.facing; // Place mario tile and keep track of where he is facing

            // Calculate in-game statistics relevant to reward function
            
            int g = 0; // Count of how many enemies were stomped
            for (MarioEvent e : this.gameEvents) {
                if (e.getEventType() == EventType.STOMP_KILL.getValue()) {
                    g += 1;
                }
            }
        	float x = world.mario.x; // x-position
            float vx = world.mario.xa; // instantaneous velocity 
            int d = world.mario.alive ? 0 : -15; // penalty for dying           
            float f = model.getCompletionPercentage() == 1? 15 : 0; // reward for full completion
            int c = 4; // assume difference in clock is 4 frames due to 4-frame skip
            
            // Agent is rewarded for moving to the right, while not dying, and doing so as quick as possible
            float reward = vx + d + f - c; // NOTE: NOT NEEDED ANYMORE, reward calculated in python from info dict

            // episode is complete if we get to the castle, die, or run out of time
            boolean done = world.gameStatus ==  GameStatus.WIN || world.gameStatus == GameStatus.LOSE || world.currentTimer <= 0;

            
            // Print subsequent lines to positionData.txt
            try{
                FileWriter myWriter = new FileWriter("src/positionData.txt", true); // TODO: Refactor so we dont open and close every timestep?
                myWriter.write(marioCoordinates);
                myWriter.close();
            }
            catch(IOException ex){
                System.out.println("An error occured");
                return new StepResult(obs, reward, done, new float[] {vx, d, f, c, x, g});
            }      
            
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
            
        return new StepResult(obs, reward, done, new float[] {vx, d, f, c, x, g});

    }

    public void killGame() {
        // this.window.dispatchEvent(new WindowEvent(window, WindowEvent.WINDOW_CLOSING));
        this.window.dispose();
    }

}

