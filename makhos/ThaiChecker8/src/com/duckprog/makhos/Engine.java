package com.duckprog.makhos;
import java.util.*;
public class Engine{
		public   boolean stoprun = false;
		int checker=100;
		int king=200;
		int edge=10;
		int random_wieght=10;
		int score;
		public int Evalution(int[][] board)
		{
			score=0;
			for(int i=0;i<8;i++)
				for(int j=0;j<8;j++)
				{
					if(board[i][j]==MainActivity.white)
					{
						score-=checker;
						score-=j*j;
					}
					else if(board[i][j]==MainActivity.whiteKing)
					{
						score-=king;
						if(i==0 || i==7)
							score+=edge;
						if(j==0 || j==7)
							score+=edge;
					}
					else if(board[i][j]==MainActivity.blackKing)
					{
						score+=king;
						if(i==0 || i==7)
							score-=edge;
						if(j==0 || j==7)
							score-=edge;
					}
					else if(board[i][j]==MainActivity.black)
					{
						score+=checker;
						score+=(7-j)*(7-j);
					}
				}
			/*for(int i=0;i<Move.initList.size();i++)
			{
				int[] intArrays=new int[2];
				intArrays=(int[])Move.initList.elementAt(i);

					if(board[intArrays[0]][intArrays[1]]==MainActivity.white)
					{
						score-=checker;
						score-=intArrays[1]*intArrays[1];
					}
					else if(board[intArrays[0]][intArrays[1]]==MainActivity.whiteKing)
					{
						score-=king;
						if(intArrays[0]==0 || intArrays[0]==7)
							score+=edge;
						if(intArrays[1]==0 || intArrays[1]==7)
							score+=edge;
					}
					else if(board[intArrays[0]][intArrays[1]]==MainActivity.blackKing)
					{
						score+=king;
						if(intArrays[0]==0 || intArrays[0]==7)
							score-=edge;
						if(intArrays[1]==0 || intArrays[1]==7)
							score-=edge;
					}
					else if(board[intArrays[0]][intArrays[1]]==MainActivity.black)
					{
						score+=checker;
						score+=(7-intArrays[1])*(7-intArrays[1]);
					}
			}*/
			score+=(int)(Math.random()*10);
			return score;
		}
		int opponent(int turn)
		{
			switch(turn)
			{
			case MainActivity.black:
			case MainActivity.blackKing:
				return MainActivity.white;
			case MainActivity.white:
			case MainActivity.whiteKing:
				return MainActivity.black;
			}
			return MainActivity.empty;
		}
		int whichTurn(int turn)
		{
			return turn==MainActivity.black?-999999:999999;
		}
		public int MiniMax(int[][] board,int depth,int maxDepth,int[] theMove,int toMove)
		{
			int x =  MiniMax(board,depth,maxDepth,theMove,toMove,999999,-999999);
			//System.out.println("step exit minimax");
			MainActivity.threadEnd = true;
			//this.stoprun = false;
			return x;
		}
		
		int MiniMax(int[][] board,int depth,int maxDepth,int[] theMove,int turn,int whiteBest,int blackBest)
		{
			int theScore;
			int[][] newBoard=new int[8][8];
			int[] bestMove=new int[4];
			Vector<int[]> movesList=new Vector<int[]>();
			int bestScore;			
			/*if (this.stoprun){
				//System.out.println("step stop run minimax");
				depth = maxDepth;
			}	*/		
			
			if(depth==maxDepth || this.stoprun)
			{
			    bestScore=this.Evalution(board);
			}
			else
			{
				movesList=Move.generateMove(board,turn);
				bestScore=whichTurn(Move.color(turn)); 
				switch(movesList.size())
				{
					case 0:
						return bestScore;
					case 1:
						if(depth==0)
						{
							bestMove=(int[])movesList.elementAt(0);
							for(int k=0;k<4;k++)
								theMove[k]=bestMove[k];
							return 0;
						}
						else
						{
							maxDepth+=1;
						}	
				}
				for(int i=0;i<movesList.size();i++)
				{
					newBoard=copyBoard(board);
					Move.moveBoard(newBoard,(int[])movesList.elementAt(i),false);
					int temp[]=new int[4];

					theScore=MiniMax(newBoard,depth+1,maxDepth,temp,opponent(turn),whiteBest,blackBest);
					if(Move.color(turn)==MainActivity.black && theScore>bestScore)
					{
						bestMove=(int[])movesList.elementAt(i);
						bestScore=theScore;
						if(bestScore>blackBest)
						{
							if(bestScore>=whiteBest)
								break;
							else
								blackBest=bestScore;
						}
					}
					else if(Move.color(turn)==MainActivity.white && theScore<bestScore)
					{
						bestMove=(int[])movesList.elementAt(i);
						bestScore=theScore;
						if(bestScore<whiteBest)
						{
							if(bestScore<=blackBest)
								break;
							else
								whiteBest=bestScore;
						}
					}
				}
			}// end else
			if((depth==0)&&(bestMove[0]==0)&&(bestMove[1]==0)&&(bestMove[2]==0)&&(bestMove[3]==0))
			{
				bestMove=(int[])movesList.elementAt(0);
				for(int k=0;k<4;k++)
					theMove[k]=bestMove[k];
				
				return 0;
			}
			else
			{
				for(int k=0;k<4;k++)
					theMove[k]=bestMove[k];
				return bestScore;
			}//*/
		}
		int[][] copyBoard(int[][] board)
		{
			int[][] copy=new int[8][8];
			for(int i=0;i<8;i++)
				System.arraycopy(board[i],0,copy[i],0,8);
			return copy;
		}

}
